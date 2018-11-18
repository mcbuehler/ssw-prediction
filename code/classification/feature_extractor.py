import numpy as np

from data_manager import DataManager
from dataset import DatapointKey as DK


class FeatureExtractor:
    """
    Class that extracts features from a given h5 file.
    """
    def __init__(self, data_file):
        self.data_manager = DataManager(data_file)

    def hist(self, variable, n_bins, hist_range=None):
        """
        Creates histogram features for given variable
        :param variable: e.g. wind_65
        :param n_bins: number of bins for histogram
        :param hist_range: min and max value for histograms.
        Use values from training set
        :return: np.array
        """
        data = self.data_manager.get_data_for_variable(variable)
        if hist_range is None:
            hist_range = (np.min(data), np.max(data))
        hist = np.apply_along_axis(
            lambda a: np.histogram(a, bins=n_bins, range=hist_range)[0],
            axis=1, arr=data)
        return hist

    def yearly_label(self, label_identifier):
        """
        Returns the yearly labels encoded as 0 and 1s
        :param label_identifier: e.g. "CP07"
        :return: np.array
        """
        data = self.data_manager.get_data_for_variable(label_identifier)
        binary_label = np.apply_along_axis(lambda a: 1 if 1 in a else 0,
                                           axis=1, arr=data)
        return binary_label


if __name__ == "__main__":
    import h5py

    data = h5py.File("../data/labeled_output/data_preprocessed_labeled.h5",
                     'r')
    extractor = FeatureExtractor(data)
    print(extractor.hist(DK.TEMP_60_70, n_bins=10).shape)
    print(extractor.yearly_label(DK.CP07))
