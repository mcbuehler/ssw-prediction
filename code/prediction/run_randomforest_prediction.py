from prediction.randomforest_hist_prediction import RandomForestPrediction
from preprocessing.dataset import DatapointKey as DK
import os


def run_experiment_grid_prediction():
    n_bins = 20
    n_estimators = 2#10000
    input_path_sim = os.getenv("DSLAB_CLIMATE_LABELED_DATA")

    definitions = [DK.CP07, DK.UT, DK.U65]
    cutoff_points = [90, 120]
    feature_intervals = [7, 14, 21, 28]
    prediction_intervals = [1, 2, 3, 4]

    # Evaluate for simulated data only
    for definition in definitions:
        for cutoff in cutoff_points:
            for feature in feature_intervals:
                for prediction in prediction_intervals:
                    print("Evaluating for "
                          "definition {}, "
                          "cutoff {}, "
                          "feature {}, "
                          "prediction {}".format(
                        definition,
                        cutoff,
                        feature,
                        prediction)
                    )
                    model = RandomForestPrediction(
                        definition=definition,
                        path=input_path_sim,
                        n_bins=n_bins,
                        n_estimators=n_estimators,
                        cutoff_point=cutoff,
                        features_interval=feature,
                        prediction_weeks=prediction
                    )

                    model.evaluate(plot=False)


if __name__ == '__main__':
    run_experiment_grid_prediction()
