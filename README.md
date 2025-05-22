# HON2200-Prosjekt-2-Gruppe-H

This project provides code for doing keyword analysis of pdfs. Our purpose is to analyse AI-strategies, but the code is designed to be flexible and can be easily modified. Our findings and accompanying analysis is detailed in our report "Demokrati og risikovillighet i nasjonale KI-strategier: En tekstanalytisk sammenlikning på tvers av regimetyper".

## Structure

#### `requirements.txt`
Contains all requirements to run the programs.

### `Code/`
Contains code for the analysis.

#### `pdf_reader.ipynb`
Notebook for converting pdf's to txt using OCR (tesseract).

- **`pdf/`**
  - Source document pdf's yet to be converted go in the main folder. Pdf's already read may be moved to the **`read/`** subfolder to avoid rereading.

- **`txt/`**
  - Contains the converted txt-files by OCR.

#### `democracy_index.xlsx`
Excel sheet with Economist democracy index from UN-website for calculating mean democracy index of EU/EEA.

#### `keywords.xlsx`
List of keywords in excel table.

#### `text_analysis.py`
Keyword analysis.

#### `Topography.py`
Module for embedding analysis inspired by Toubia et al., 2021.

#### `embedding.ipynb`
Embedding analysis for sylistic comparisons and clustering.

#### `embedding.pkl`
Pickle file with embeddings.

#### `embeddings.csv`
Stored embedding data.



### `Report/`
Contains the written report and supplementary material.

#### `Demokrati_og_risikovillighet_i_nasjonale_KI-strategier.pdf`
The report in pdf format.

- **`Figures/`**
  - Contains all figures used in the report and some tables with results.

- **`Additional_plots/`**
  - Contains additonal figures and plots not included in the report.


## Dataset

The data is sourced from OECD:

- OECD. (2025). AI in China. Hentet Mai 2025 fra OECD.AI: https://oecd.ai/en/dashboards/countries/China

- OECD. (2025). AI in Poland. Hentet Mai 2025 fra OECD.AI: https://oecd.ai/en/dashboards/countries/Poland

- OECD. (2025). AI in the European Union. Hentet Mai 2025 fra OECD.AI: https://oecd.ai/en/dashboards/countries/EuropeanUnion

- OECD. (2022, Januar 13). AI in the United States. Hentet fra OECD.AI: https://oecd.ai/en/dashboards/countries/UnitedStates


The project is based on example 6, alternative 2 in the description at https://henriasv.github.io/hon2200-v25/oppgaver/prosjekt_2.html.