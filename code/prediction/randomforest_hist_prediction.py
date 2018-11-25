import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, \
    accuracy_score
from sklearn.model_selection import cross_val_score

from preprocessing.dataset import DatapointKey as DK
import os
import numpy as np
import matplotlib.pyplot as ply
from prediction.base_model import PredictionBaseModel


np.random.seed(42)


class RandomForestPrediction(PredictionBaseModel):
    """A class that receives as input the processed data and the definition that
    you want prediction for and does prediction using the RandomForest
    Classifier.
    """

    def __init__(self, definition, path, n_bins=100, n_estimators=100):
        """
        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
                e.g. "CP07"
            path: string
                the path where the preprocessed input data are
            n_bins: int
                number of bins to use for the histogram extraction
            n_estimators: int
                number of estimators to use in the RandomForestClassifier

        """
        super().__init__(definition, path)
        self.n_bins = n_bins

        self.metrics = [f1_score, roc_auc_score, accuracy_score]
        self.classifier = RandomForestClassifier(n_estimators=n_estimators)

        self.X = None
        self.y = None

        self._prepare()

    def _histograms(self, data, n_bins, hist_range=None):
        """
        Creates histogram features for given variables.
        For each variable, we create a histogram with n_bins.

        Parameters
        ----------
            data: numpy.array
                contains the preprocessed data
                of shape (num_data, num_variables, num_days)
            n_bins: int
                number of bins to use in the histograms

        """
        if hist_range is None:
            # TODO: fix this not to use test data
            hist_range = (np.min(data), np.max(data))
        hist = np.apply_along_axis(
            lambda a: np.histogram(a, bins=n_bins, range=hist_range)[0],
            axis=2, arr=data)
        return hist

    def _produce_features(self, data):
        """
        Gets the data in the format [num_data, num_variables, num_days]
        and produces the design matrix X.

        Returns
        -------
            X: np.array
                A numpy array of size
                [num_data, num_features]
        """
        histogram_features = self._histograms(data, n_bins=self.n_bins)
        X = np.reshape(histogram_features, (histogram_features.shape[0], -1))
        return X

    def _produce_labels(self, labels):
        """
        Gets the data in the format [num_data, num_labels] and outputs
        the ready-to-use labels for prediction.

        Returns
        -------
            labels: np.array
                A numpy array of size
                [num_data,]
        """
        return labels.ravel()

    def evaluate(self, plot=False):
        """
        Trains the model in a 5-fold cross validation and returns a mean
        and standard deviation for each metric used for evaluation.

         Parameters
        ----------
            plot: bool
                Whether to show a plot or not.

        Returns
        -------
            mean_scores: list of length len(self.metrics),
            std_scores: list of length len(self.metrics)
        """
        assert self.ready

        scores = [
            cross_val_score(self.classifier, self.X, self.y, cv=5,
                            scoring=make_scorer(metric))
            for metric in self.metrics
        ]

        scores_means = [np.mean(score) for score in scores]
        scores_2std = [2 * np.std(score) for score in scores]

        if plot:
            self.plot(scores_means, scores_2std)
        return scores_means, scores_2std

    def plot(self, scores_mean, scores_std):
        """
        Creates a bar plot with error bars for given means and std.
        One bar corresponds to one metric.
        For the error bars, use 2*std.
        At a later stage, we can use bootstrapping to estimate a correct
        confidence interval.

        Parameters
        ----------
            scores_mean: list of length len(self.metrics) with scores
            for each metric.
            scores_std: list of length len(self.metrics) with
            standard deviations for each metric
        """
        # Make sure to have the correct order for classifier_txt
        classifiers_txt = ["F1", "ROCAUC", "Accuracy"]

        ply.figure()
        ply.bar(classifiers_txt, scores_mean, yerr=scores_std, align='center',
                alpha=0.5, ecolor='black', capsize=10)

        ply.title(
            "Results {}".format(self.__class__.__name__))
        # TODO: save fig
        # ply.savefig(os.path.join(path_plots, "{}.png".format(variable)))
        ply.ylim(0, 1)

        ply.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple prediction \
            scheme using histogram features and RandomForest')
    parser.add_argument(
            "-d",
            "--definition",
            choices=(DK.CP07, DK.UT, DK.U65),
            help="Choose the definition that you want to run classification",
            action="store",
            default=DK.CP07
           )
    parser.add_argument(
            "-i",
            "--input_path",
            help="Choose the input relative path where the data are",
            action="store",
            default=os.getenv("DSLAB_CLIMATE_LABELED_DATA")
            )
    args = parser.parse_args()

    model = RandomForestPrediction(
            definition=args.definition,
            path=args.input_path
            )
    model.evaluate(plot=True)




