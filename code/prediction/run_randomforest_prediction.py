import logging

from sklearn.exceptions import UndefinedMetricWarning

from prediction.randomforest_hist_prediction import RandomForestPrediction
from preprocessing.dataset import DatapointKey as DK
import os


def run_experiment_grid_prediction():
    n_bins = 20
    n_estimators = 1000
    input_path_sim = os.getenv("DSLAB_CLIMATE_LABELED_DATA")

    definitions = [DK.CP07, DK.UT, DK.U65]
    cutoff_points = [90, 120]
    feature_intervals = [7, 14, 21, 28]
    prediction_intervals = [7, 14, 21, 28]
    prediction_start_days = [0, 7, 14, 21]

    # Evaluate for simulated data only
    for definition in definitions:
        for cutoff in cutoff_points:
            for feature in feature_intervals:
                for prediction in prediction_intervals:
                    for start_day in prediction_start_days:
                        print("Evaluating for "
                              "definition {}, "
                              "cutoff {}, "
                              "feature {}, "
                              "prediction {},"
                              "start day {}".format(
                                definition,
                                cutoff,
                                feature,
                                prediction,
                                start_day))

                        model = RandomForestPrediction(
                            definition=definition,
                            path=input_path_sim,
                            n_bins=n_bins,
                            n_estimators=n_estimators,
                            cutoff_point=cutoff,
                            features_interval=feature,
                            prediction_start_day=start_day,
                            prediction_interval=prediction
                        )
                        model.evaluate(plot=False)

if __name__ == '__main__':
    run_experiment_grid_prediction()
