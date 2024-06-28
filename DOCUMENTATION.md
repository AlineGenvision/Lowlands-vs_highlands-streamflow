# Repository Documentation

This file contains a high-level overview of the repository structure and the role of each file within. Files and functions are usually commented. Notebooks include additional explanation as well.

If you find any bugs in the code or have trouble setting up your environment, please contact me directly! My email is linked in the README!

-----

## Environment Setup

To run the project locally, it is recommended you set up your environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

1. Clone this repository using `git clone --recursive <HTTPS or SSH>` to ensure the apollo submodule is loaded as well.
2. Navigate to the newly created repository directory (i.e., use `cd Lowlands-vs-highlands-streamflow`).
3. Create your `conda` Python environment using `conda env create -n environment_name `.
4Install all the packages with `pip install -r requirements.txt`.

-----

## Repository Overview

Many directories containn an `archive/` sub-directory, which contains code that was produced during the project, but is not needed to reproduce our final analysis. Many of these archived files are well-commented or self-explanatory, but given their secondary nature we do not describe them with the same level of detail in this overview.

-----

### `main directory/`

The main directory contains two Python scripts used for data download and preprocessing.

- `assembly.py`: Downloads all the needed data from ERA5, aggregates it into daily values and performs the linear / surface interpolation.
- `assembly_HR.py`: Downloads all the needed data from ERA5-land (higher resolution data), aggregates it into daily values and performs the linear / surface interpolation. This might take a while.

-----

### `preprocessing/`

This directory contains all the files needed to process the data. Some files contain helper functions for the 'assembly.py' and 'assembly_HR.py' scripts, others halp with visualisation and organisation of the data.

-----

- `catchment_characteristics.py`: Helps to create an overview of all the catchments defined in 'data/Catchments_Data', with their characteristics such as latitude, slope, hydrological efficiency etc.
- `surface_interpolation.py`: Contains all the functions and helper functions (removing NaNs etc) needed to interpolated over a gridded dataset with a certain resolution.
- `visualisation.py`: Collection of functions to create the visualisations in the notebooks (compare datasets, plot them on a yearly basis, plot performances in relationship to catchment characteristics etc.)

### `train_model/`

This directory contains all the files needed to train the NN models and evaluate them. 

-----

- `load_data.py`: Loads the data, split it into training and validation and formats the input needed for PyTorch.
- `RE_utils.py`: Calculates specific error metric (Reflective Error).
- `sensitivity analysis.py`: Performs sensitivity analysis given a network and an input dataset. Alters the distribution on the inputs slightly and reports the result on the model outputs.
- `train.py`: AnteCedent NN network. Contains all the information  needed to train the network with grid search or normally. All input parameters such as features, dropout, learning rate etc. are input parameters.

-----

### `notebooks/`

This directory contains all of our major data processing and all modelling experiments.

**NN Exploration:**
- `ANN_Replication.ipynb/`: Replicating the Feedforward NN from previous work [1], testing it on various catchments and plotting it together.
- `ANN Expansion.ipynb/`: Expansion of the ANN Replication, including hourly data points and snow melt values.
- `NRFA_vs_ERA5.ipynb/`: Uses the NN approach to compare the performance from two different input datasets, also has some visualisation to assess the data bias.

**Data Visualization:**
- `Data_visualisation.ipynb/`: Contains all more in-depth plots giving an overview of all the 26 catchments across the UK.

**Surface Interpolation:**
- `Surface Interpolation.ipynb`: Walks step-by-step through the interpolation (with a lot of visualisation in between) for 'surface interpolation.py'. Tutorial Notebook.
- `Surface Interpolation_HighRes.ipynb`: Similar to the normal 'Surface Interpolation.ipynb' notebook, but focuses on the higher resolution data.

**Gaussian Processes:**
- `Gaussian Processes.ipynb`: Tutorial Notebook on SSVGP's. Uses similar data processing as the NN and trains GPs on them with the GPJax framework.

-----

## Reproducing Report Figures and Tables

See the tables below for the notebooks to run to reproduce each figure and table in the final report. 

### Figures

| Figure # |               Notebook                |
|:--------:|:-------------------------------------:|
|    1     |      `Data_visualisation.ipynb`       |
|    2     |                   -                   |
|    3     |   `Data_visualisation.ipynb.ipynb`    |
|    4     |        `ANN_replication.ipynb`        |
|    5     |        `ANN_replication.ipynb`        |
|    6     |         `ANN_expansion.ipynb`         |
|    7     |         `ANN_expansion.ipynb`         |
|    8     |         `ANN_expansion.ipynb`         |
|    9     |         `NRFA_vs_ERA5.ipynb`          |
|    10    |         `NRFA_vs_ERA5.ipynb`          |
|    11    |         `NRFA_vs_ERA5.ipynb`          |
|    12    |     `Surface Interpolation.ipynb`     |
|    13    |     `Surface Interpolation.ipynb`     |
|    14    | `Surface Interpolation_HighRes.ipynb` |
|    15    | `Surface Interpolation_HighRes.ipynb` |
|    16    | `Surface Interpolation_HighRes.ipynb` |
|    17    |      `Gaussian Processes.ipynb`       |
|    18    |      `Gaussian Processes.ipynb`       |
|    19    |      `Gaussian Processes.ipynb`       |
|    20    |      `Data_visualisation.ipynb`       |
|    21    |      `Data_visualisation.ipynb`       |
|    22    |     `Surface Interpolation.ipynb`     |
|    23    |      `Gaussian Processes.ipynb`       |
|    24    |         `NRFA_vs_ERA5.ipynb`          |

### Tables

**Note:** as a general case, one cannot obtain the full results needed to reproduce each table with one run of the notebook. Instead, the user must change clearly marked hyperparameters, input variables, and preprocessing steps to sequentially obtain all results.

| Table # |                 Notebook                 |
|:-------:|:----------------------------------------:|
|    1    |         `ANN_replication.ipynb`          |
|    2    |          `ANN_expansion.ipynb`           |
|    3    |          `ANN_expansion.ipynb`           |
|    4    |           `NRFA_vs_ERA5.ipynb`           |
|    5    |        `Gaussian Processes.ipynb`        |
|    6    |                    -                     |
|    7    |                    -                     |
|    8    |                    -                     |
|    9    |                    -                     |
|  
# References
[1] Robert Edwin Rouse, Doran Khamis, Scott Hosking, Allan McRobie, and Emily Shuckburgh. “Streamflow Prediction Using Artificial Neural Networks & Soil Moisture Proxies”. Unpublished manuscript. Unpublished manuscript., Oct. 2023.