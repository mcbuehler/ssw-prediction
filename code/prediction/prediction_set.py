import numpy as np
from core.data_manager import DataManager
from preprocessing.dataset import DatapointKey as DK


class PredictionSetBase:
    """A class that receives as input the processed variables and the definition
    that you want classification for and returns the data either in the correct
    format for prediction (either using a fixed cut-off point or a sliding
    window approach. It also sets a prediction interval in the future where you
    will do classification for
    """

    def __init__(self, definition, path, cutoff_point, prediction_start_day,
                 prediction_interval):
        """The constructor of the PredictionSet class

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
                e.g. "CP07"
            path: string
                the path where the input data are
                e.g. "../data/data_labeled.h5"
            cutoff_point: int
                the maximum cutoff point until where your data will expand
            prediction_start_day: int
                the day where you want to make predictions for after the
                cutoff_point
            prediction_interval: int
                the interval where you will make predictions for
        """

        self.definition = definition
        self.path = path
        self.cutoff_point = cutoff_point
        self.prediction_start_day = prediction_start_day
        self.prediction_interval = prediction_interval
        self.data_manager = DataManager(self.path)

    def get_labels(self):
        """Returns the binary labels as a numpy array for the corresponding
        definition for the prediction interval set.
        Label meaning: "There is a SSW in the timeframe
        [cutoff point, cutoff point + prediction interval]"

        Returns
        -------
            labels: numpy array
                A numpy array of size [num_data x 1] with the labels
        """

        temp_labels = self.data_manager.get_data_for_variable(self.definition)
        labels = np.zeros((temp_labels.shape[0], 1))
        # returns 1 if there is an SSW from the cutoff point day until the
        # prediction interval day
        for i in range(len(temp_labels)):
            labels[i, 0] = int(np.any(
                temp_labels[
                    i,
                    self.cutoff_point + self.prediction_start_day:
                    self.cutoff_point + self.prediction_start_day +
                    self.prediction_interval
                ]))
        return labels

    def get_features(self):
        """Returns the data as a numpy array.
        Returns
        -------
            data: numpy array
        """
        raise NotImplementedError("Use a subclass")


class CutoffWindowPredictionSet(PredictionSetBase):
    def __init__(self, definition, path, cutoff_point, prediction_start_day,
                 prediction_interval):
        """The constructor of the PredictionSet class

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
            path: string
                the path where the input data are
            cutoff_point: int
                the maximum cutoff point until where your data will expand
            prediction_start_day: int
                the day where you want to make predictions for after the
                cutoff_point
            prediction_interval: int
                the interval where you will make predictions for
        """
        super().__init__(definition, path, cutoff_point, prediction_start_day,
                         prediction_interval)

    def get_features(self):
        """Returns the data as a numpy array for the variables wind_60,
        wind_65,  temp_60_90 until the cutoff point

        Returns
        -------
            data: numpy array
                A numpy array of size [num_data, num_features, cutoff_point]
        """
        # data comes in the format (N, FC, D)
        data = self.data_manager.get_data_for_variables([DK.WIND_60,
                                                         DK.WIND_65,
                                                         DK.TEMP_60_90])

        return data[:, :, :self.cutoff_point]


class FixedWindowPredictionSet(PredictionSetBase):
    def __init__(self, definition, path, cutoff_point, prediction_start_day,
                 prediction_interval, feature_interval):
        """The constructor of the PredictionSet class

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
            path: string
                the path where the input data are
            cutoff_point: int
                the maximum cutoff point until where your data will expand
            prediction_start_day: int
                the day where you want to make predictions for after the
                cutoff_point
            prediction_interval: int
                the interval where you will make predictions for
            feature_interval: int
                the interval where you will extract features from
        """
        super().__init__(definition, path, cutoff_point, prediction_start_day,
                         prediction_interval)

        # We need to make sure that we don't access negative indices
        assert (self.cutoff_point - feature_interval) >= 0
        self.feature_interval = feature_interval

    def get_features(self):
        """Returns the data as a numpy array for the variables wind_60,
        wind_65, temp_60_90 from cutoff point - feature_window_size
        until the cutoff point.
        Let's say we use a feature_interval of 30 days and set the
        cutoff point to day 60.
        Then, this function will return features for days 30-60.

        Returns
        -------
            data: numpy array
                A numpy array of size [num_data, num_features, cutoff_point]
        """
        # data comes in the format
        # (number_samples, number_features, number_days)
        data = self.data_manager.get_data_for_variables([DK.WIND_60,
                                                         DK.WIND_65,
                                                         DK.TEMP_60_90])
        # We do not consider data from days earlier than start_index
        start_index = self.cutoff_point - self.feature_interval
        return data[:, :, start_index:self.cutoff_point]


if __name__ == '__main__':
    import os

    cutoff_point = 70
    prediction_start_day = 7
    prediction_interval = 7
    path_input = os.getenv("DSLAB_CLIMATE_LABELED_DATA")
    target = DK.CP07

    # Test cutoff window where we take all data from t=0 to t=cutoff_point
    cutoff_prediction_set = CutoffWindowPredictionSet(
        target,
        path_input,
        cutoff_point,
        prediction_start_day,
        prediction_interval
    )
    features1 = cutoff_prediction_set.get_features()
    labels1 = cutoff_prediction_set.get_labels()
    print(features1.shape, labels1.shape)

    # Test cutoff window where we take data from
    #  t=cutoff_point-feature_interval to t=cutoff_point
    feature_interval = 60
    fixed_prediction_set = FixedWindowPredictionSet(
        target,
        path_input,
        cutoff_point,
        prediction_start_day,
        prediction_interval,
        feature_interval
    )

    features2 = fixed_prediction_set.get_features()
    labels2 = fixed_prediction_set.get_labels()
    print(features2.shape, labels2.shape)
