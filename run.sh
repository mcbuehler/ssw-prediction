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
    for cutoff in 60 90
    do
        for feature in 30 40 50
        do
            for prediction in 5 10 15 20 25 30
            do
                for data in real sim
                do
                    python code/prediction/xgboost_prediction.py -d $definition -cp $cutoff -fi $feature -pi $prediction -dt $data&
                done
            done
            wait
        done
    done
done
