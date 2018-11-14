import numpy as np
import h5py

class DataManager:

    def __init__(self, data_file_path):
        self.data_file = h5py.File(data_file_path, 'r')
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

    def yearly_label(self, label_identifier):
        data = self._get_data_for_variable(label_identifier)
        binary_label = np.apply_along_axis(lambda a: 1 if 1 in a else 0, axis=1, arr=data)
        return binary_label


if __name__ == "__main__":
    extractor = DataManager("../data/labeled_output/data_preprocessed_labeled.h5", 'r')
    print(len(extractor.yearly_label("CP07")))
    print(extractor._get_data_for_variable("CP07").shape)
