import numpy as np

from prediction.prediction_set import FixedWindowPredictionSet

np.random.seed(42)


class PredictionBaseModel:
    """Inherit from this class to create your own PredictionModels.
    """

    def __init__(self, definition, path, cutoff_point=90,
                 prediction_interval=30, features_interval=30, cv_folds=5):
        """
        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
                e.g. "CP07"
            path: string
                the path where the preprocessed input data are
        """
        self.path = path
        self.definition = definition

        self.cv_folds = cv_folds

        self.cutoff_point = cutoff_point
        self.prediction_interval = prediction_interval
        self.features_interval = features_interval

        self.prediction_set = self._get_prediction_set()

    def _get_prediction_set(self):
        """
        Runs all the computations to get this instance ready for evaluation.
        For example, you can initialise self.X and self.y here.
        """
        prediction_set = FixedWindowPredictionSet(
            definition=self.definition,
            path=self.path,
            cutoff_point=self.cutoff_point,
            prediction_interval=self.prediction_interval,
            feature_interval=self.features_interval
        )
        return prediction_set

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
        return NotImplementedError("Implement this method in a subclass")

    def plot(self, scores_mean, scores_std):
        """
        Creates a bar plot with error bars for given means and std.
        One bar corresponds to one metric.

        Parameters
        ----------
            scores_mean: list of length len(self.metrics) with scores
            for each metric.
            scores_std: list of length len(self.metrics) with
            standard deviations for each metric
        """
        return NotImplementedError("Implement this method in a subclass")
