import argparse

from imblearn.over_sampling import ADASYN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, \
    accuracy_score
from sklearn.model_selection import StratifiedKFold, \
    cross_validate
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

from preprocessing.dataset import DatapointKey as DK
from utils.enums import Classifier, Task, Metric, DataType
import os
import numpy as np
import matplotlib.pyplot as ply
from prediction.base_model import PredictionBaseModel

from utils.set_seed import SetSeed
from utils.dslab_logging import get_logger
from utils.output_class import Output


SetSeed().set_seed()
logger = get_logger()


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

    def _histograms(self, data, n_bins):
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
        hist = np.apply_along_axis(
            lambda a: np.histogram(a, bins=n_bins)[0],
            axis=2, arr=data)
        return hist

    def transform(self, X, y=None, **fit_params):
        histogram_features = self._histograms(X, n_bins=self.n_bins)
        X = np.reshape(histogram_features, (histogram_features.shape[0], -1))
        return X

    def fit(self, X, y=None, **fit_params):
        # No fitting required
        return self


class RandomForestPrediction(PredictionBaseModel):
    """A class that receives as input the processed data and the definition that
    you want prediction for and does prediction using the RandomForest
    Classifier.
    """

    def __init__(self,
                 definition,
                 path,
                 cutoff_point=90,
                 prediction_interval=7,
                 prediction_start_day=7,
                 features_interval=7,
                 cv_folds=5,
                 n_bins=100,
                 n_estimators=100):
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
        super().__init__(definition, path, cutoff_point=cutoff_point,
                         prediction_interval=prediction_interval,
                         prediction_start_day=prediction_start_day,
                         features_interval=features_interval,
                         cv_folds=cv_folds)
        self.n_bins = n_bins

        self.metrics = [f1_score, roc_auc_score, accuracy_score]
        self.metric_txt = ["F1", "ROCAUC", "Accuracy"]

        self.classifier = RandomForestClassifier(n_estimators=n_estimators)

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

        # Get the raw data for the features
        X = self.prediction_set.get_features()
        # Bring labels in correct format
        y = self.prediction_set.get_labels().ravel()

        # We have an unbalanced dataset, so we stratify
        cv = StratifiedKFold(self.cv_folds, shuffle=True)

        logger.info("Scoring for {} metrics...".format(len(self.metrics)))

        # We keep scores in a dict instead of a list to have more semantics
        scores = {txt: list() for txt in self.metric_txt}

        for train_index, test_index in cv.split(X, y):
            # Create train / test splits
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Extract histogram features
            X_train = feature_extraction.fit_transform(X_train)
            X_test = feature_extraction.fit_transform(X_test)

            # We only resample the training data
            X_train, y_train = self._resample(X_train, y_train)

            # Train model
            model = self.classifier
            model.fit(X_train, y_train)

            # Evaluate model
            pred = model.predict(X_test)
            for i, metric in enumerate(self.metrics):
                score = metric(y_test, pred)
                scores[self.metric_txt[i]].append(score)

        print(scores)
        scores = [scores[txt] for txt in self.metric_txt]


        # We only want to keep mean and std for each metric
        scores_means = [np.mean(score) for score in scores]
        scores_std = [np.std(score) for score in scores]

        if plot:
            self.plot(scores_means, scores_std)

        # Write results to output file
        output_default_args = dict(
            classifier=Classifier.randomforest,
            task=Task.prediction,
            data_type=DataType.simulated,
            definition=self.definition,
            cutoff_point=self.cutoff_point,
            feature_interval=self.features_interval,
            prediction_start_day=self.prediction_start_day,
            prediction_interval=self.prediction_interval
        )

        out_metrics = (
            (Metric.f1, scores[0]),
            (Metric.auroc, scores[1]),
            (Metric.accuracy, scores[2])
        )
        for metric, score in out_metrics:
            Output(
                **output_default_args,
                metric=metric,
                scores=score
            ).write_output()

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
        path=args.input_path,
        n_bins=20,
        n_estimators=100,
        cutoff_point=90,
        features_interval=21,
        prediction_interval=14,
        prediction_start_day=7
    )

    print(model.evaluate(plot=False))
