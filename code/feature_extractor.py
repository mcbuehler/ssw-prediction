import numpy as np
from dataset import DatapointKey as DK


class FeatureExtractor:

    def __init__(self, data_file):
        self.data_file = data_file
        self.data_dict = dict()

    def _get_np_array(self, variable):
        """

        :param variable:
        :return: matrix nxd where n is number of instances and d is number of dimensions
        """
        a = np.stack([self.data_file[key][variable] for key in list(self.data_file.keys())])
        return a

    def _get_data_for_variable(self, variable):
        if variable not in self.data_dict:
            self.data_dict[variable] = self._get_np_array(variable)
        return self.data_dict[variable]

    def mean(self, variable):
        data = self._get_data_for_variable(variable)
        return np.mean(data, axis=1)

    def median(self, variable):
        data = self._get_data_for_variable(variable)
        return np.median(data, axis=1)

    def std(self, variable):
        data = self._get_data_for_variable(variable)
        return np.std(data, axis=1)

    def hist(self, variable, n_bins, hist_range=None):
        data = self._get_data_for_variable(variable)
        if hist_range is None:
            hist_range = (np.min(data), np.max(data))
        hist = np.apply_along_axis(lambda a: np.histogram(a, bins=n_bins, range=hist_range)[0], axis=1, arr=data)
        return hist

    def yearly_label(self, label_identifier):
        data = self._get_data_for_variable(label_identifier)
        binary_label = np.apply_along_axis(lambda a: 1 if 1 in a else 0, axis=1, arr=data)
        return binary_label



if __name__ == "__main__":
    import h5py
    data = h5py.File("../data/labeled_output/data_preprocessed_labeled.h5", 'r')

    extractor = FeatureExtractor(data)
    print(extractor.mean(DK.TEMP_60_70).shape)
    print(extractor.hist(DK.TEMP_60_70, n_bins=10).shape)
    print(extractor.yearly_label(DK.CP07))
