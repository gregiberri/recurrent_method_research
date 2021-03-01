# Recurrent Method Research

## Set up

#### conda env
To set up the conda environment use the <i>docker/config_files/recurrent.yaml</i> file:
```shell script
conda env create -f docker/config_files/recurrent.yaml
```

#### docker
Currently docker usage is not supported.

#### data
IRMAS data should be downloaded from [here](https://zenodo.org/record/1290750#.YD01EHVKhH5) and then should be unzipped.

## Usage
To use the code run the <i>run.py</i> file with the argument stating which config file to use 
(default: <i>config/config_files/irmas_all.yaml</i>)

Set the data.params.path to the folder of your IRMAS data.

The code exports the results to the following path: <i>../results/recurrent_method_research/{id}</i>