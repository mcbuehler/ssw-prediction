from prediction.prediction_set import FixedWindowPredictionSet


class PredictionBaseModel:
    """Inherit from this class to create your own PredictionModels.
    """

    def __init__(self, definition, path, cutoff_point=90,
                 prediction_weeks=2, features_interval=30, cv_folds=5):

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
        self.prediction_weeks = prediction_weeks
        self.features_interval = features_interval

        self.prediction_set = self._get_prediction_set()

    def _get_prediction_set(self):
        prediction_set = FixedWindowPredictionSet(
            definition=self.definition,
            path=self.path,
            cutoff_point=self.cutoff_point,
            prediction_interval=self.prediction_weeks,
            feature_interval=self.features_interval
        )
        return prediction_set

    def _produce_features(self, data):
        """
        Gets the data in the format [num_data, num_variables, len_winter]
        and produces a matrix [num_data,num_features].

        Returns
        -------
            features: np.array
                A numpy array of size
                [num_data, num_features]
        """
        return NotImplementedError("Implement this method in a subclass")

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
        return NotImplementedError("Implement this method in a subclass")

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
