"""
Dette er en modul jeg skrev til et annet prosjekt. Vi bruker veldig lite av funksjonaliteten, 
men det er raskere å importere klassen for å gjøre analysen enn å kopiere over de relevante delene.
(Den bør altså ikke regnes som en del av dette prosjektet. Se heller på anvendelsen av den i embedding.ipynb)
"""


# Standard imports
import io
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy as sc

# Special imports
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from PIL import Image
 
# Importing the transformer
from sentence_transformers import SentenceTransformer

"""
Change the default model here. To save the model locally, replace the path with your path and run this module.
At the end of your path, add the name of the folder you want create for the model, eg. ".../local_model"
When you have downloaded the model, uncomment model_name = path to use the stored model.
"""
model_name = "mixedbread-ai/mxbai-embed-large-v1" 
path = None # replace with local path
#path = "C:/Users/jonas/OneDrive/Dokumenter/Python Scripts/embed/local_model_sentence_transformers" # (example path / for my convenience)
#model_name = path # Uncomment this line once you have downloaded the model.

# Standard metric
l2 = lambda x, y: np.linalg.norm(y-x)

# Helpers
def _read_chunks(filename, chunksize=25):
    """
    Reads text into a list of strings with the specified number of words (discards final chunk to ensure similar length).
    """
    chunks = []

    with open(filename, 'r', encoding='utf-8') as infile:
        buffer = []
        wordcount = 0

        for line in infile:
            words = line.split()
            buffer.extend(words)
            wordcount += len(words)

            while wordcount >= chunksize:
                chunk = buffer[:chunksize]
                buffer = buffer[chunksize:]
                wordcount -= chunksize

                chunk = ' '.join(chunk) # Casting to string
                chunks.append(chunk)

    return chunks


def _tsp(embeddings, scale=1e6, Test=False):
    """
    Solves the traveling salesman problem with fixed start and endpoints approximately for the embeddings.
    
    Input is a N x d matrix of N points in d dimensions, and optionally a scaling-factor which determines the decimal-precision of the distance matrix. 
    This factor is only important if the points are very close. In such a case, it could be greatly increased without problems (I hope).
    Output is the minimum required distance. If Test is True, the route is also printed, together with the cumulative distance which should be equal to the return value.
    
    We use Google OR-Tools to solve TSP. The following code is from the OR-Tools TSP-page, only slightly modified for our purposes.
    """
    
    N, d = np.shape(embeddings)
    # Calculating and scaling the distance matrix. Rounding is to illustrate that the solver only uses the integer-part of the values, but is not nescessary.
    M = (sc.spatial.distance_matrix(embeddings,embeddings)*scale).round()

    manager = pywrapcp.RoutingIndexManager(N, 1, [0], [N-1]) # number of points, number of cars, start index, end index
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return round(M[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        if Test:
            index = routing.Start(0)
            plan_output = "Route:\n"
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += f" {manager.IndexToNode(index)} ->"
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            plan_output += f" {manager.IndexToNode(index)}\n"
            plan_output += f"Total distance: {route_distance/scale}\n"
            print(plan_output)
        return solution.ObjectiveValue()/scale


def _HDBSCAN_clustering(x_2d):
    """
    Uses HDBSCAN do do unsupervised clustering of the points.
    
    Input is an array of points in 2d.
    Output is a list with the color of each point in 2d.
    """
    from sklearn.cluster import HDBSCAN

    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=5) # Initializing the HDBSCAN model
    clusters = hdbscan.fit_predict(x_2d) # Using HBDSCAN to cluster the points

    # Finding the number of clusters:
    if -1 in clusters:
        num_colors = len(set(clusters)) - 1 # -1 (no cluster) should not be included in the count of clusters
    else:
        num_colors = len(set(clusters))

    colormap = matplotlib.colormaps['gist_rainbow'] # Specifying the colormap
    grey = (0.0, 0.0, 0.0, 1) # Specifying the the color for the points not in a cluster

    norm = mcolors.Normalize(vmin=0, vmax=num_colors) # The cluster indexes must be normalized in order to represent a color

    # Assigning a color to each point based on the cluster indexes
    color_list = []
    for cluster_index in clusters:
        if cluster_index == -1: 
            color_list.append(grey) # The points not in a cluster are black
        else:
            color_list.append(colormap(norm(cluster_index)))
    
    return color_list



class Topography:
    """
    A base class for topographic analysis of texts inspired by Toubia et al., 2021:
    Embed texts chunkwise or with a custom function and save them, or load pre-embedded texts directly.
    Calculate speed/circuitousness and volume. Note that all return-values from analysis are log-transformed.
    Visulaise the semantic path in 2D with plot or animate.
    """
    def __init__(self, filename, table=None, con=None, model=model_name, metric=l2):
        """
        Input is normally the path to a text file you wish to chunkwise embed/analyze.
        
        You may choose any model from sentence_transformers, and you can load a local model by letting model be the path.

        If a table and a connnection to a sql-database is provided (use: con = sqlite3.connect("database_name.db")),
        filename should be the filename of the pre-embedded text in the given table in that database. 
        
        You may also change the metric for testing differences - it should take two vectors and output their separation.
        """
        self.filename = filename
        self.metric = metric
        self.model = SentenceTransformer(model)
        if table:
            assert con, "You must provide a connection to an sql-database to load files."
            c = con.cursor()
            c.execute(f'SELECT embedding FROM {table} WHERE filename=?', (filename,))
            blob = c.fetchone()[0]
            self.embeddings = pickle.loads(blob)
            con.close()


    def embed(self, chunksize=25, reader=_read_chunks):
        """
        Returns the embeddings in an array.
        
        The default reader reads chunksize number of words at a time. 
        You may use a different reader, so long as it takes self.filename and chunksize as arguments.
        """
        model = self.model

        # Reads chunks
        chunks = reader(self.filename,chunksize)
        self.chunks = chunks
        
        # Embeds the chunks
        embeddings = []
        for chunk in chunks:
            embeddings.append(model.encode(chunk))
        
        assert len(embeddings) > 1, "Failed to get embeddings: You might have provided a short text or a bad reader."
        self.embeddings = np.asarray(embeddings)
    

    def save_embeddings(self,path):
        """
        Saves the embeded text to a .npy-file at the path specified.
        """
        assert hasattr(self,"embeddings"), "No embeddings to save."
        np.save(path, self.embeddings)


    def speed(self,scale=1e6):
        """
        Calculates speed (average local euclidian distance) and circuitousness (speed / minimum required speed).
        
        Scale is a scaling-factor for TSP which determines the decimal-precision of the distance matrix. 
        This factor is only important if the points are very close. In such a case, it could be greatly increased without problems (I hope).
        """
        assert hasattr(self,"embeddings"), "No embeddings to analyze."
        embeddings = self.embeddings

        T = len(embeddings)
        distances = np.zeros(T-1)
        for t in range(T-1):
            distances[t] = self.metric(embeddings[t],embeddings[t+1])
            speed = sum(distances)/(T-1)
        
        self.distances = distances
        minimum_required_speed = _tsp(embeddings,scale)/(T-1)
        circuitousness = speed/minimum_required_speed

        return np.log(speed), np.log(circuitousness)
    
    
    def variance(self,trim=0):
        """
        Returns the variance of the distances. If trim is a non-zero percentage (less than 50), we do
        a trim% trim of the distances and return the greatest difference in the remaining sample.
        (Used for stability-testing and to measure the effect of preprocessing.)
        """
        assert hasattr(self,"distances"), "Please run the speed-method before running variance."
        distances = self.distances

        if not trim:
            return np.var(distances) # Population variance
        
        assert 0 < trim < 50, "Invalid trim."

        lower_bound = np.percentile(distances,trim)
        upper_bound = np.percentile(distances,100-trim)
        return np.ptp(distances[(distances >= lower_bound) & (distances <= upper_bound)])
    

    def volume(self,tol=.01):
        """
        Calculates minimum volume ellipsoid containing all N embeddings in the N-1 dimensional subspace defined by the embedding-vectors.
        Since different texts may have different numbers of points, we normalize over dimensions by taking the geometric mean of the axes of the ellipsoid.
        (This is representative of the volume since the factor of enlargement of the unit hypersphere is the inverse of the square root of the determinant of 
        the centre form ellipsoid matrix, or equivalently the product of the inverse of the square roots of the eigenvalues, which are the lengths of the axes.
        Still, this seems not to be quite a succesful normalization in the extremes where N is close either to 0 or d.)

        Input is a N x d matrix of N embeddings in d dimensions, and optionally a specified toleance which should be smaller than 1.
        Output is normalized volume as defined above, which should be a measure of the ground covered by the chunkwise embedded text.

        We solve the dual problem as described in Moshtagh, 2005. This implementation is based on his matlab-code.
        """
        assert hasattr(self,"embeddings"), "No embeddings to analyze."
        P = self.embeddings # Points along each row
        N, d = P.shape

        if N <= d: # We find a distance-preserving subspace of N-1 dimensions in which all points reside to get a nonzero volume ellipoid. 
            # Centering points about last point (translations preserve relative distances)
            centre = P[-1]
            tmp = P - centre
            Y = tmp[:-1].T # Final point is the origin and should be excluded to get T-1 dimensions. Transpose for mathematical convention.

            # Finding an orthonormal basis
            U, S, Vt = np.linalg.svd(Y, full_matrices=False) # SVD, note that U (d x d) is orthogonal and preserves distances.
            basis = U[:, :N-1] # (We assume with probability ~1 that the embeddings are linearly independent.)

            # Finding the coordinate vectors of the embeddings in the subspace
            transform = basis.T @ Y 
            P = np.hstack([transform,np.zeros((N-1,1))]) # Adding back the origin
        else: 
            """
            With 25-word chunks and 300 (1024) dimentions, this would be a 7500 (25600) word text. 
            It is non-trivial that case is directly comparable to the subspace case, where we have N-1 dimensions.
            Thus, we should check this if we use datasets containing both texts exceeding and subceeding this threshold.
            """
            P = P.T # Mathematical convention.
        
        # --- Finding centre form ellipsoid matrix ---
        d, N = np.shape(P) # d <= N-1
        Q = np.vstack([P, np.ones(N)]) # The dual problem is explained in Moshtagh, 2006

        # Initializations
        count = 0
        err = 1.
        u = 1./N * np.ones(N) # 1st iteration / "feasible point"

        # Khachiyan algorithm
        while err > tol:
            X = Q @ np.diag(u) @ Q.T # Since u is 1D, diag returns a 2D array with u on the diagonal.
            M = np.diag(Q.T @ np.linalg.inv(X) @ Q)

            j = np.argmax(M)
            maximum = M[j]
            stepsize = (maximum - d - 1)/((d + 1)*(maximum - 1))
            new_u = (1 - stepsize)*u
            new_u[j] += stepsize

            count += 1
            err = np.linalg.norm(new_u - u)
            u = new_u

        # Ellipse parameters in centre form
        U = np.diag(u)
        c = P @ u
        A = (1./d)*np.linalg.inv(P @ U @ P.T - np.outer(c,c))

        # Normalized volume
        eigenvals = np.linalg.eigvals(A)
        axes = eigenvals**(-.5)
        logvolume = sum(np.log(axes))/(N-1) # log(geometric mean)

        return logvolume
    
    
    def _reduce_dimensions(self, perplexity):
        """
        Helper. Uses t-SNE to reduce the dimensionality of the embeddings.
        Output is an array of points in 2D.
        """
        assert hasattr(self,"embeddings"), "No embeddings to analyze."
        embeddings = self.embeddings

        assert perplexity < len(embeddings), 'The perplexity must be lower than the number of chunks.'

        # Importing and initializing the t-SNE model
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=42) 

        # Reducing to 2D
        return tsne.fit_transform(embeddings)


    def plot(self, perplexity=10):
        """
        Uses t-SNE to visualise the embeddings in 2D.
        Uses HDBSCAN do do unsupervised clustering of the points.
        Plots the points in different colors according to the clusters.
        """
        x_2d = self._reduce_dimensions(perplexity) # t-SNE reduction
        color_list = _HDBSCAN_clustering(x_2d)

        offset = np.ptp(x_2d[:, 0])/120 # Calculating an offset to the points to scale the labels properly

        # 2D visualisation with enumerated points
        plt.figure(figsize=(8, 6)) 
        for i, point in enumerate(x_2d):
            plt.scatter(point[0], point[1], label=f"Point {i}", color=color_list[i], edgecolor='none') # Plotting the points with colours
            plt.text(point[0] + offset, point[1] + offset, f" {i}", fontsize=6)  # Plotting number labels with offset

        plt.title('t-SNE visualization with enumerated points')
        plt.show()


    def animate(self, destination, perplexity=10):
        """
        Uses t-SNE to visualise the embeddings in 2D.
        Uses HDBSCAN do do unsupervised clustering of the points.
        Makes an animation of the path through the points.

        The resulting gif is saved at the path specified by destination.
        """

        x_2d = self._reduce_dimensions(perplexity) # t-SNE reduction
        color_list = _HDBSCAN_clustering(x_2d)

        x = x_2d[:, 0]
        y = x_2d[:, 1]
        x_range = np.ptp(x)
        y_range = np.ptp(y)

        axis_set = [min(x) - x_range/20, max(x) + x_range/20, min(y) - y_range/20, max(y) + y_range/20] # Calculating suitable axis
        plt.figure(figsize=(8, 6)) # Fixing the size of the plot
        plt.axis(axis_set) # Fixing the axis of the plot
        plt.title('t-SNE animation') # Setting the title of the plot

        frames = []
        for i, point in enumerate(x_2d):
            plt.scatter(point[0], point[1], label=f"Point {i}", color=color_list[i]) # Plotting the points

            if i > 0:
                x_values = [x[i-1], x[i]]
                y_values = [y[i-1], y[i]]
                plt.plot(x_values, y_values, color='grey', lw=1) # Plotting lines between the points
            
            # Saving frames as images to IO-stream
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(Image.open(buf))

        # Saving frames as GIF
        frames[0].save(destination, save_all=True, append_images=frames[1:], duration=400, loop=0)
        plt.close()


    def store_model(self, path=path):
        """
        Creates a folder storing the model at the given path.
        """
        if os.path.exists(path):
            print("This folder already exists. Make sure to append the name of the folder you wish to create where the model will be stored.")

        else:
            os.makedirs(path)
        
            model = self.model
            model.save(path)
            print(f"The model was successfully stored at {path}.")



if __name__ == "__main__" and path:
        if input(f"Type 'y' if you want to store the model at {path}: ") == 'y':
            text = Topography(None)
            text.store_model()
        else:
            print("Download aborted.")