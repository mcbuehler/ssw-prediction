import argparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, \
    accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from core.data_manager import DataManager
from preprocessing.dataset import DatapointKey as DK
import os
import numpy as np
import matplotlib.pyplot as ply

np.random.seed(42)


class HistogramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=100):
        """
        Transformer that produces histogram features

        Parameters
        ----------
            n_bins: int
                number of bins to use in the histograms
        """
        self.n_bins = n_bins

    def _histograms_for_variable(self, array, n_bins):
        """
        Computes the histograms for array of shape (n_samples, n_days)

        Parameters
        ----------
            array: np.array
                array of shape (n_samples, n_days)
            n_bins: int
                number of bins to use in the histograms
        """
        # Compute the histogram range over all given samples in array
        hist_range = [np.min(array), np.max(array)]

        histograms = np.apply_along_axis(
            lambda a: np.histogram(a, bins=n_bins, range=hist_range)[0],
            axis=1,
            arr=array)
        return histograms

    def _histograms(self, data, n_bins=100):
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
        n_features = data.shape[1]
        # For each feature variable, compute the histograms for all samples
        result = np.array([
            self._histograms_for_variable(data[:, i, :], n_bins=n_bins)
            for i in range(n_features)
        ])
        # result has shape (n_features, n_samples, n_bins)
        # We reshape it to have the shape (n_samples, n_features, n_bins)
        result = np.swapaxes(result, 0, 1)
        return result

    def transform(self, X, y=None, **fit_params):
        """
        Extracts histogram features from X.

        Parameters
        ----------
            X: np.array
                array of shape (n_samples, n_features, n_days)
            y: None
        """

        histogram_features = self._histograms(X, n_bins=self.n_bins)
        # We need to reshape the array of shape (n_samples, n_features, n_bins)
        # to (n_samples, n_features * n_bins)
        X = np.reshape(histogram_features, (histogram_features.shape[0], -1))
        return X

    def fit(self, X, y=None, **fit_params):
        # No fitting required
        return self


class RandomForestClassification():
    """A class that receives as input the processed data and the definition that
    you want prediction for and does prediction using the RandomForest
    Classifier.
    """

    def __init__(self, definition, path, n_bins=100, n_estimators=100,
                 cv_folds=5):
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
        self.data_manager = DataManager(path)

        self.definition = definition

        self.cv_folds = cv_folds
        self.n_bins = n_bins

        self.metrics = [f1_score, roc_auc_score, accuracy_score]
        self.classifier = RandomForestClassifier(n_estimators=n_estimators)

    def _get_raw_data(self):
        data = self.data_manager.get_data_for_variables(
            [
                DK.TEMP_60_90,
                DK.WIND_60,
                DK.WIND_65
            ]
        )
        return data

    def _get_labels(self):
        raw_labels = self.data_manager.get_data_for_variable(self.definition)
        binary_label = np.apply_along_axis(lambda a: 1 if 1 in a else 0,
                                           axis=1, arr=raw_labels)
        return binary_label

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
        # We create a pipeline in order to apply the same independent
        # preprocessing steps to all folds in the cross-validation
        feature_extraction = HistogramTransformer(n_bins=self.n_bins)
        steps = [
            ('feature_extraction', feature_extraction),
            ('model', self.classifier)
        ]
        pipeline = Pipeline(steps)

        # Get the raw data for the features
        raw_data = self._get_raw_data()
        # Bring labels in correct format
        labels = self._get_labels()

        # We have an unbalanced dataset, so we stratify
        cv = StratifiedKFold(self.cv_folds, shuffle=True)

        # Produce scores for all scoring metrics
        scores = [
            cross_val_score(pipeline, raw_data, labels, cv=cv,
                            scoring=make_scorer(metric))
            for metric in self.metrics
        ]

        # We only want to keep mean and std for each metric
        scores_means = [np.mean(score) for score in scores]
        scores_std = [np.std(score) for score in scores]

        if plot:
            self.plot(scores_means, scores_std)

        return scores_means, scores_std

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
        x_txt = ["F1", "ROCAUC", "Accuracy"]

        ply.figure()
        ply.bar(x_txt, scores_mean, yerr=scores_std, align='center',
                alpha=0.5, ecolor='black', capsize=10)

        ply.title(
            "Results {}".format(self.__class__.__name__))

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

    print("n_bins --> scores for F1, ROCAUC, Accuracy")
    for n_bins in [5, 10, 20, 30, 50, 80, 120, 150, 200]:
        model = RandomForestClassification(
            definition=args.definition,
            path=args.input_path,
            n_bins=n_bins,
            n_estimators=1000
        )

        mean_scores, _ = model.evaluate(plot=False)
        print("n_bins: {} --> {:.4f}, {:.4f}, {:.4f}".format(n_bins,
                                                             *mean_scores))
