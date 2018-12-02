#!/bin/bash

for cutoff in 60 90
do
    for definition in CP07 U\&T U65
    do
        for feature in 30 40 50
        do
            for prediction in 5 10 15 20 25 30
            do
                #echo "Definition:$definition-Cutoff:$cutoff-Prediction:$prediction"
                python code/prediction/xgboost_prediction.py -d $definition -cp $cutoff -pi $prediction -fi $feature >> results.txt& 
            done 
            wait
        done
    done
done
