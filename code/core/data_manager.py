import numpy as np
import h5py


class DataManager:
    """
    Class that facilitates loading data from the h5 File
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

    def get_data_for_variables(self, variables):
        """
        Returns a 3D numpy array for a number of variables
        :param variables:
        :return: np.array
        """
        # TODO: Check that this function is working properly
        data = np.array(list(self.get_data_for_variable(v) for v in variables))
        # Feature Count, Number of samples, Number of Days
        data = np.swapaxes(data, 0, 1)
        return data


if __name__ == "__main__":
    dm = DataManager(
        "../data/labeled_output/data_preprocessed_labeled.h5")
    print(dm.get_data_for_variable("CP07").shape)
