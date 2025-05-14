# HON2200-Prosjekt-2-Gruppe-H

This project provides code for doing keyword analysis of pdfs. Our purpose is to analyse AI-strategies, but the code is designed to be flexible and can be easily modified. Our findings and accompanying analysis is detailed in our report at XXX.

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


#### `keywords.xlsx`
List of keywords in excel table.

#### `text_analysis.py`
Keyword analysis.

#### `Topography.py`
Module for embedding analysis inspired by Toubia et al., 2021.

#### `embedding.ipynb`
Embedding analysis for sylistic comparisons.

#### `embeddings.csv`
Stored embedding data.



### `Report/`
Contains the written report and supplementary material.

#### `Report title`
The report in pdf format

- **`Figures/`**
  - Contains all figures used in the report.

- **`Additional_plots/`**
  - Contains additonal figures and plots not included in the report.

