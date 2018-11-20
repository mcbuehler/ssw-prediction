import numpy as np
from core.data_manager import DataManager


class PredictionSet:
    def __init__(self, definition, path, cutoff_point, max_prediction):
        self.definition = definition
        self.path = path
        self.cutoff_point = cutoff_point
        self.prediction_interval = max_prediction
        self.data_manager = DataManager(self.path)

    def _get_labels_for_prediction(self):
        temp_labels = self.data_manager.get_data_for_variable(self.definition)
        labels = np.zeros((temp_labels.shape[0], 1))
        # returns 1 if there is an SSW from the cutoff point day until the
        # prediction interval day
        for i in range(len(temp_labels)):
            labels[0, i] = int(np.any(temp_labels[i, self.cutoff_point:
                                                  self.cutoff_point +
                                                  self.prediction_interval]))
        return labels
