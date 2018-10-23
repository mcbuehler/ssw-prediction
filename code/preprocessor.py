import numpy as np
import argparse
import pickle
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from bisect import bisect_right
# import sys


class Preprocessor:
    USEFUL_PRES = 10
    START_DAY = 270
    END_DAY = 120

    def __init__(self, first_winter_name, second_winter_name, min_pres=6,
                 max_pres=16, int_pres_low=0, int_pres_high=2):
        self.first_winter = Dataset(first_winter_name)
        self.second_winter = Dataset(second_winter_name)

    def get_pres_interpol(self, search_variable, search_value):
        """Finds in an array the indices of the values that have to be
        interpolated using binary search.

        Parameters
        ----------
            search_variable: string
                The variable that we have to search in (e.g. latitude).
            search_value: int
                The value of the variable that needs to be searched.

        Returns
        -------
             idx_before: int
                The index of the array before the value of interest.
             idx_after: int
                The index of the array after the value of interest.

        """
        var_to_search = self.first_winter[search_variable][:].data
        try:
            idx_after = bisect_right(var_to_search, search_value)
            idx_before = idx_after - 1
        except IndexError:
            print("Cannot interpolate in the given model")

        return idx_before, idx_after

    def get_useful_part(self, variable):
        """Finds in an array the indices of the values that have to be
        interpolated using binary search.

        Parameters
        ----------
            variable: string
                The variable where we have to extract the useful part.

        Returns
        -------
             winter: numpy array
                The winter after the preprocessing that is common has been done
                with dimensions (180, 2, 64).

        """
        # get the winter part from the two years and concatenate them
        first_winter_part = self.first_winter[variable][self.START_DAY:]
        second_winter_part = self.second_winter[variable][:self.END_DAY]
        winter = np.concatenate((first_winter_part, second_winter_part),
                                axis=0)
        # get the useful pressure levels (~10-~100 hPa)
        min_pres, _ = self.get_pres_interpol('pfull', self.USEFUL_PRES)
        _, max_pres = self.get_pres_interpol('pfull', 100)
        winter = winter[:, min_pres:max_pres + 1, :, :]
        # average through longitudes
        winter = np.mean(winter, axis=3)
        # get the necessary pressure levels (before and after 10 hPa)
        int_pres_low, int_pres_high = self.get_pres_interpol('pfull',
                                                             self.USEFUL_PRES)
        winter = winter[:, int_pres_low:int_pres_high + 1, :]
        return winter

    def get_uwind(self, lat):
        """Get the uwind at a specific latitude.

        Parameters
        ----------
            lat: int
                The latitude that you want to compute the wind at.

        Returns
        -------
             winter_intrep: numpy array
                The final calculated values with dimensions (180, 1).

        """
        wind = self.get_useful_part('ucomp')
        # find the useful latitudes and pressure levels and keep only those in
        # the initial matrix
        idx_low_lat, idx_high_lat = self.get_pres_interpol('lat', lat)
        idx_low_pres, idx_high_pres = self.get_pres_interpol('pfull',
                                                             self.USEFUL_PRES)

        wind = wind[:, :, idx_low_lat:idx_high_lat + 1]
        wind_interp = np.zeros(shape=wind.shape[0])
        # loop over the days
        for i in range(len(wind_interp)):
            # calculate the x values to be interpolated for pressure and
            # latitude
            x_pres = self.first_winter.variables[
                    'pfull'][idx_low_pres:idx_high_pres + 1].data
            x_lat = self.first_winter.variables[
                    'lat'][idx_low_lat:idx_high_lat + 1].data
            y = []
            # interpolate for every latitude
            for j in range(2):
                f = interp1d(x_lat, wind[i, :, j])
                y.append(f(lat))
            # interpolate for the pressure level
            y = np.asarray(y)
            f = interp1d(x_pres, y)
            wind_interp[i] = f(self.USEFUL_PRES)

        return wind_interp

    def preprocess_temp(self, lower_lat):
        """Preprocess the temperature at a specific latitude and onwards.

        Parameters
        ----------
            lower_lat: int
                The latitude that you will start preprocessing the latitude at.

        Returns
        -------
             temp_interp: numpy array
             idx_low_lat: int

        """

        temp = self.get_useful_part('temp')
        # remove the latitudes that we don't need
        idx_low_lat, _ = self.get_pres_interpol('lat', lower_lat)
        temp = temp[:, :, idx_low_lat:]
        # keep only the pressure levels before and after 10 hPa
        idx_low_pres, idx_high_pres = self.get_pres_interpol('pfull',
                                                             self.USEFUL_PRES)
        temp_interp = np.zeros(shape=(temp.shape[0], temp.shape[2]))
        # interpolate though the pressure levels
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                x = self.first_winter.variables[
                    'pfull'][idx_low_pres:idx_high_pres + 1].data
                y = temp[i, :, j]
                f = interp1d(x, y)
                temp_interp[i, j] = f(self.USEFUL_PRES)

        return temp_interp, idx_low_lat

    def get_polar_temp(self, lower_lat, higher_lat):
        temp, idx_low_lat = self.preprocess_temp(lower_lat=lower_lat)
        # useful_lat = self.first_winter.variables['lat'][idx_low_lat:].data
        # weights = np.cos(np.deg2rad(useful_lat))

    @staticmethod
    def construct_feature_matrix(pkl_file=None):
        try:
            with open(pkl_file, 'rb') as f:
                suf_ar = pickle.load(f)
        except FileNotFoundError:
            with open(str(pkl_file), 'wb') as f:
                pickle.dump(suf_ar, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessor for the model data')
    parser.add_argument('--year1', help='Path to the first year',
                        required=True)
    parser.add_argument('--year2', help='Path to the second year',
                        required=True)
    args = parser.parse_args()

    preprocess = Preprocessor(args.year1, args.year2)
    temp = preprocess.get_polar_temp(60, 90)
    print(temp.shape)
    # wind_60 = preprocess.get_uwind(60)
    # print(wind_60)
