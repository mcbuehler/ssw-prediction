#!/bin/bash

# classification
# for definition in CP07 U\&T U65
# do
#     python code/classification/xgboost_simple.py -d $definition -m CV&
# done
# 
# for definition in CP07 U\&T U65
# do
#     python code/classification/xgboost_simple.py -d $definition -dt real&
# done

# prediction

for definition in CP07 U\&T U65
do
    for cutoff in 90 120
    do
        for feature in 7 14 21 28
        do
            for prediction in 1 2 3 4
            do
                for data in real sim
                do
                    python code/prediction/xgboost_prediction.py -d $definition -cp $cutoff -fi $feature -wi $prediction -dt $data&
                done
            done
            wait
        done
    done
done
