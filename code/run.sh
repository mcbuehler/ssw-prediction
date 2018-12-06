#!/bin/bash

for cutoff in 90 120
do
    for definition in CP07 U\&T U65
    do
        for feature in 7 14 21 28
        do
            for prediction in 1 2 3 4
            do
                #echo "Definition:$definition-Cutoff:$cutoff-Prediction:$prediction"
                python code/prediction/xgboost_prediction.py -d $definition -cp $cutoff -pi $prediction -fi $feature >> results.txt& 
            done 
            wait
        done
    done
done
