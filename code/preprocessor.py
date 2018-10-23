import numpy as np
import argparse
import pickle
from scipy.interpolate import interp2d
from netCDF4 import Dataset
from bisect import bisect_right


class Preprocessor:
    def __init__(self, first_winter_name, second_winter_name, min_pres=6,
                 max_pres=16, int_pres_low=0, int_pres_high=2):
        self.first_winter = Dataset(first_winter_name)
        self.second_winter = Dataset(second_winter_name)
        self.useful_pres = 10

    def get_pres_interpol(self, search_variable, search_value):
        """ Finds in an array the indices of the values that have to be
        interpolated using binary search
        Args:
            self: an instance of the class
            search_variable: the variable that we have to search in (e.g.
            latitude)
            search_value: the value of the variable that needs to be searched

        Returns:
             idx_before: the index of the array before the value of interest
             idx_after: the index of the array after the value of interest

        """
        var_to_search = self.first_winter[search_variable][:].data
        try:
            idx_after = bisect_right(var_to_search, search_value)
            idx_before = idx_after - 1
        except IndexError:
            print("Cannot interpolate in the given model")

        return idx_before, idx_after

    def get_useful_part(self, variable):
        # TODO: Remove hardcoded values of the winter
        first_winter_part = self.first_winter[variable][270:]
        second_winter_part = self.second_winter[variable][:120]
        winter = np.concatenate((first_winter_part, second_winter_part),
                                axis=0)
        min_pres, _ = self.get_pres_interpol('pfull', self.useful_pres)
        _, max_pres = self.get_pres_interpol('pfull', 100)
        winter = winter[:, min_pres:max_pres + 1, :, :]
        winter = np.mean(winter, axis=3)
        int_pres_low, int_pres_high = self.get_pres_interpol('pfull',
                                                             self.useful_pres)
        winter = winter[:, int_pres_low:int_pres_high + 1, :]
        return winter

    def get_uwind(self, lat):
        wind = self.get_useful_part('ucomp')
        idx_low_lat, idx_high_lat = self.get_pres_interpol('lat', lat)
        idx_low_pres, idx_high_pres = self.get_pres_interpol('pfull',
                                                             self.useful_pres)

        wind = wind[:, :, idx_low_lat:idx_high_lat + 1]
        wind_interp = np.zeros(shape=wind.shape[0])
        # loop over the days
        for i in range(len(wind_interp)):
            x = self.first_winter.variables[
                    'lat'][idx_low_lat:idx_high_lat + 1].data
            y = self.first_winter.variables[
                    'pfull'][idx_low_pres:idx_high_pres + 1].data
            z = wind[i]
            f = interp2d(x, y, z)
            wind_interp[i] = f(lat, self.useful_pres)

        return wind_interp

    def get_mean_temp_80_90(self):
        raise(NotImplementedError)

    def get_mean_temp_60_70(self):
        raise(NotImplementedError)

    @staticmethod
    def construct_feature_matrix():
        raise(NotImplementedError)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessor for the model data')
    parser.add_argument('--year1', help='Path to the first year',
                        required=True)
    parser.add_argument('--year2', help='Path to the second year',
                        required=True)
    args = parser.parse_args()

    preprocess = Preprocessor(args.year1, args.year2)
