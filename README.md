# DSL2018-Proj-Climate-Science

## Introduction
When Europeans experience a heavy temperature drop in winter,
chances are that a *Sudden Stratospheric Warming* (SSW) has
happened. This phenomenon is characterized by a strong temperature
increase in the stratosphere, possibly up to 50 ̊C within a few days.
Being able to predict these events can help meteorologists improve
weather forecasting.

Download the [poster](poster.pdf) if you want to get a quick overview about the project. If you would like more information, you can download the [report](report.pdf) or explore our code. For any feedback, don't hesitate to contact me. Thank you!

## Setup
1. Set the environment variables.
Default values for spaceml servers are given here:
`scripts/load_environment_variables.sh`

2. Copy the simulated and real data to the corresponding folders

3. Run the preprocessing (see below)



## Preprocessing

How to run:
1. Check and adjust environment variables. Default values can be loaded via 
`source scripts/load_environment_variables.sh`
2. (Optional, depends on your configuration) Load your python virtual environment:
`source YOUR_ENV/bin/activate`
3. Run the preprocessing scripts for both the simulated and real data:
`python code/run_preprocessing.py`
`python code/run_preprocessing_real.py`
4. Run the labeling script:
`python code/run_label_generation.py`

In order to run the code you have to set up your PYTHONPATH to the code folder
```export PYTHONPATH="${PYTHONPATH}:/where/the/code/folder/is/"```


## Classification

### RandomForest and Histograms

Go to the `code` folder and execute this script:
`python classification/run_randomforest_classification.py`
The results will be written to the results file defined by the environment variable `DSLAB_RESULT_FILE`

### CNNs

Go to the `code` folder and execute this script:
`python classification/cnn.py`
The results will be written to the results file defined by the environment variable `DSLAB_RESULT_FILE`

If the resulting Pytorch model weights is to be persisted, use `--savemodel` flag and set `CNN_WEIGHTS` environment variable to the directory where the model weights is going to be persisted.

### XGBoost
By going to the `code` folder and executing this script:
`python classification/xgboost_simple.py` you can see the following output:

```
usage: xgboost_simple.py [-h] [-d {CP07,U65,ZPOL_temp,U&T}]
                         [-sp SIMULATED_PATH] [-rp REAL_PATH] [-dt {sim,real}]
                         [-m {TT,CV}] [-p]

A simple classification scheme using feature engineering and the
XGBoostClassifier

optional arguments:
  -h, --help            show this help message and exit
  -d {CP07,U65,ZPOL_temp,U&T}, --definition {CP07,U65,ZPOL_temp,U&T}
                        Choose the definition that you want to run
                        classification
  -sp SIMULATED_PATH, --simulated_path SIMULATED_PATH
                        Choose the input relative path where the simulated
                        data are
  -rp REAL_PATH, --real_path REAL_PATH
                        Choose the input relative path where the real data are
  -dt {sim,real}, --data_type {sim,real}
                        Choose if the evaluation is going to happen on real
                        orsimulated data
  -m {TT,CV}, --mode {TT,CV}
                        Choose the evaluation mode
  -p, --produce_importance
                        Choose if you'll produce the feature importances
```
where you can decide on various parameters. The results will be written to the results file defined by the environment variable `DSLAB_RESULT_FILE`

## Prediction

### XGBoost + tsfresh
By going to the `code` folder and executing this script:
`python prediction/xgboost_prediction.py` you can see the following output:

```
usage: xgboost_prediction.py [-h] [-d {CP07,U65,U&T}] [-sp SIMULATED_PATH]
                             [-rp REAL_PATH] [-dt {sim,real}] [-m {TT,CV}]
                             [-cp CUTOFF_POINT] [-fi FEATURES_INTERVAL]
                             [-sd PREDICTION_START_DAY]
                             [-pi PREDICTION_INTERVAL] [-p]

A prediction scheme using feature engineering and the XGBoostClassifier

optional arguments:
  -h, --help            show this help message and exit
  -d {CP07,U65,U&T}, --definition {CP07,U65,U&T}
                        Choose the definition that you want to run
                        classification
  -sp SIMULATED_PATH, --simulated_path SIMULATED_PATH
                        Choose the input relative path where the simulated
                        data are
  -rp REAL_PATH, --real_path REAL_PATH
                        Choose the input relative path where the real data are
  -dt {sim,real}, --data_type {sim,real}
                        Choose if the evaluation is going to happen on real
                        orsimulated data
  -m {TT,CV}, --mode {TT,CV}
                        Choose the evaluation mode
  -cp CUTOFF_POINT, --cutoff_point CUTOFF_POINT
                        Choose the cutoff point of the time series
  -fi FEATURES_INTERVAL, --features_interval FEATURES_INTERVAL
                        Choose the interval where you will calculate features
  -sd PREDICTION_START_DAY, --prediction_start_day PREDICTION_START_DAY
                        Choose the day you will start making predictions for
  -pi PREDICTION_INTERVAL, --prediction_interval PREDICTION_INTERVAL
                        Choose the interval you are going to make predictions
                        for
  -p, --produce_importance
                        Choose if you'll produce the feature importances
```

where you can decide on various parameters. The results will be written to the results file defined by the environment variable `DSLAB_RESULT_FILE`

### XGBoost + autoenconders

By going to the `code` folder and executing this script:
`python prediction/xgboost_prediction_autoencoders.py` you can see the following output:

```
usage: xgboost_prediction_autoencoders.py [-h] [-d {CP07,U65,U&T}]
                                          [-sp SIMULATED_PATH] [-rp REAL_PATH]
                                          [-dt {sim,real}] [-m {TT,CV}]
                                          [-cp CUTOFF_POINT]
                                          [-fi FEATURES_INTERVAL]
                                          [-sd PREDICTION_START_DAY]
                                          [-pi PREDICTION_INTERVAL] [-n] [-s]

A prediction scheme using feature engineering and the XGBoostClassifier

optional arguments:
  -h, --help            show this help message and exit
  -d {CP07,U65,U&T}, --definition {CP07,U65,U&T}
                        Choose the definition that you want to run
                        classification
  -sp SIMULATED_PATH, --simulated_path SIMULATED_PATH
                        Choose the input relative path where the simulated
                        data are
  -rp REAL_PATH, --real_path REAL_PATH
                        Choose the input relative path where the real data are
  -dt {sim,real}, --data_type {sim,real}
                        Choose if the evaluation is going to happen on real
                        orsimulated data
  -m {TT,CV}, --mode {TT,CV}
                        Choose the evaluation mode
  -cp CUTOFF_POINT, --cutoff_point CUTOFF_POINT
                        Choose the cutoff point of the time series
  -fi FEATURES_INTERVAL, --features_interval FEATURES_INTERVAL
                        Choose the interval where you will calculate features
  -sd PREDICTION_START_DAY, --prediction_start_day PREDICTION_START_DAY
                        Choose the day you will start making predictions for
  -pi PREDICTION_INTERVAL, --prediction_interval PREDICTION_INTERVAL
                        Choose the interval you are going to make predictions
                        for
  -n, --denoising       Choose if you are going to train the denoising version
  -s, --scale           Choose if you are going to scale the features
```

where you can decide on various parameters. The results will be written to the results file defined by the environment variable `DSLAB_RESULT_FILE`
