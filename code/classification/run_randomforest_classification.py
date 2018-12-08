from classification.randomforest_hist_classification import \
    RandomForestClassification
from preprocessing.dataset import DatapointKey as DK
import os


def run_experiment_grid_classification():
    n_bins = 20
    n_estimators = 10000
    input_path_sim = os.getenv("DSLAB_CLIMATE_LABELED_DATA")
    input_path_real = os.getenv("DSLAB_CLIMATE_LABELED_REAL_DATA")

    definitions = [DK.CP07, DK.UT, DK.U65]

    # Evaluate real data
    for definition in definitions:
        model = RandomForestClassification(
            definition=definition,
            path_train=input_path_sim,
            n_bins=n_bins,
            n_estimators=n_estimators,
            path_test=input_path_real
        )
        model.evaluate_real()
        print("Evaluated real for definition {}".format(definition))

        model = RandomForestClassification(
            definition=definition,
            path_train=input_path_sim,
            n_bins=n_bins,
            n_estimators=n_estimators
        )
        model.evaluate_simulated()
        print("Evaluated simulated for definition {}".format(definition))


if __name__ == '__main__':
    run_experiment_grid_classification()
