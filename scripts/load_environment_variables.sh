#!/usr/bin/env bash
export DSLAB_CLIMATE_BASE_INPUT=/mnt/ds3lab-scratch/dslab2018/bernatj/model
export DSLAB_CLIMATE_BASE_OUTPUT=/mnt/ds3lab-scratch/climate/preprocessed_output
export DSLAB_CLIMATE_LABELED_DATA=/mnt/ds3lab-scratch/climate/preprocessed_output/data_labeled.h5
export DSLAB_CLIMATE_PLOTS=/mnt/ds3lab-scratch/climate/plots/
export DSLAB_N_JOBS=12
export DSLAB_CLEAR_PREVIOUS=0
export DSLAB_LIMIT=-1

export DSLAB_CLIMATE_BASE_INPUT_REAL=/mnt/ds3lab-scratch/dslab2018/bernatj/reanalysis/JRA-55
export DSLAB_CLIMATE_BASE_OUTPUT_REAL=/mnt/ds3lab-scratch/climate/preprocessed_real_output
export DSLAB_CLIMATE_LABELED_REAL_DATA=/mnt/ds3lab-scratch/climate/preprocessed_real_output/data_labeled.h5

export DSLAB_RESULT_FILE=/mnt/ds3lab-scratch/climate/results/results.csv