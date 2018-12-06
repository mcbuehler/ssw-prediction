# DSL2018-Proj-Climate-Science

## Setup
1. Set the environment variables.
Default values for spaceml servers are given here:
`scripts/load_environment_variables.sh`


## Preprocessing

How to run:
1. Check and adjust environment variables. Default values can be loaded via 
`source scripts/load_environment_variables.sh`
2. (Optional, depends on your configuration) Load your python virtual environment:
`source YOUR_ENV/bin/activate`
3. Run the preprocessing script:
`python code/run_preprocessing.py`
4. Run the labeling script:
`python code/run_label_generation.py`

In order to run the code you have to set up your PYTHONPATH to the code folder
```export PYTHONPATH="${PYTHONPATH}:/where/the/code/folder/is/"```
In order to run the jupyter notebooks you have to set up your jupyter path to the code folder
```export JUPYTER_PATH="${JUPYTER_PATH}:/where/the/code/folder/is/"```


## Classification

### RandomForest and Histograms

Go to the `code` folder and execute this script: 
`python classification/run_randomforest_classification.py`
The results will be written to the results file defined by the environment variable `DSLAB_RESULT_FILE`