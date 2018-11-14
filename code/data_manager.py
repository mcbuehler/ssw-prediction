import numpy as np
import h5py


class DataManager:
    """
    Class that faciliates loading data from the h5 File
    """

    def __init__(self, data_file_path):
        """
        :param data_file_path: Path including file name to h5 File
        """
        self.data_file = h5py.File(data_file_path, 'r')
        # self.data_dict will store the data in the format
        # key: variable -> value: np.array
        self.data_dict = dict()

    def _get_np_array(self, variable):
        """
        Returns an np array with the data for given variable
        :param variable: e.g. wind_65
        :return: matrix nxd where n is number of instances and d is
         number of dimensions
        """
        a = np.stack([self.data_file[key][variable] for key in
                      list(self.data_file.keys())])
        return a

    def get_data_for_variable(self, variable):
        """
        Efficient implementation to returns an np array
        with data for given variable.
        If the data hasn't been used yet, it will be stored in self.data_dict
        :param variable:
        :return: np.array
        """
        if variable not in self.data_dict:
            # We have not loaded that data yet.
            self.data_dict[variable] = self._get_np_array(variable)
        return self.data_dict[variable]

    def yearly_label(self, label_identifier):
        """
        Returns the yearly labels for given definition (label_identifier)
        :param label_identifier:
        :return:
        """
        data = self.get_data_for_variable(label_identifier)
        binary_label = np.apply_along_axis(lambda a: 1 if 1 in a else 0,
                                           axis=1, arr=data)
        return binary_label


if __name__ == "__main__":
    dm = DataManager(
        "../data/labeled_output/data_preprocessed_labeled.h5")
    print(len(dm.yearly_label("CP07")))
    print(dm.get_data_for_variable("CP07").shape)
