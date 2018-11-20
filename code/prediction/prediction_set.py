import numpy as np
from core.data_manager import DataManager
from preprocessing.dataset import DatapointKey as DK


class PredictionSet:
    """A class that receives as input the processed variables and the definition
    that you want classification for and returns the data either in the correct
    format for prediction (either using a fixed cut-off point or a sliding
    window approach. It also sets a prediction interval in the future where you
    will do classification for
    """

    def __init__(self, definition, path, cutoff_point, max_prediction):
        """The constructor of the PredictionSet class

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
            path: string
                the path where the input data are
            cutoff_point: int
                the maximum cutoff point until where your data will expand
            prediction_interval: int
                the interval where you will do predictions in
        """

        self.definition = definition
        self.path = path
        self.cutoff_point = cutoff_point
        self.prediction_interval = max_prediction
        self.data_manager = DataManager(self.path)

    def get_labels_for_prediction(self):
        """Returns the binary labels as a numpy array for the corresponding
        definition for the prediction interval set

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
            labels[i, 0] = int(np.any(temp_labels[i, self.cutoff_point:
                                                  self.cutoff_point +
                                                  self.prediction_interval]))
        return labels

    def cutoff_for_prediction(self):
        """Returns the data as a numpy array for the variables wind_60, wind_65,
        temp_60_90 until the cutoff point

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

    def sliding_window_for_prediction(self, step, window_size, start_point):
        """Returns the data as a numpy array for the variables wind_60, wind_65,
        temp_60_90 using a sliding window size approach starting from the start
        point with a step until it reaches the cutoff point

        Returns
        -------
            data: numpy array
                A numpy array of size [num_data*window_fit, num_features,
                window_size]
        """

        # initialize all the different variables
        initial_data = self.data_manager.get_data_for_variables([DK.WIND_60,
                                                                 DK.WIND_65,
                                                                 DK.TEMP_60_90]
                                                                )
        # expand the data points until you have  reached the cutoff point by
        # moving by step size and having as length the window size
        data = np.empty((0, initial_data.shape[1], window_size))
        day_index = start_point
        while day_index + window_size <= self.cutoff_point:
            data = np.concatenate((data, initial_data[:, :, day_index:day_index
                                   + window_size]), axis=0)
            day_index += step

        data = np.array(data)
        return data


if __name__ == '__main__':
    test = PredictionSet(
                DK.CP07,
                'data/data_labeled.h5',
                60,
                15)

    # temp = test.get_labels_for_prediction()
    # print(temp.shape)
    # temp = test.cutoff_for_prediction()
    # print(temp.shape)
    temp = test.sliding_window_for_prediction(5, 30, 25)
    print(temp.shape)
