import numpy as np
import argparse
from scipy.interpolate import interp1d
from netCDF4 import Dataset


class Preprocessor:
    def __init__(self, first_winter_name, second_winter_name, min_pres=6,
                 max_pres=16, int_pres_low=0, int_pres_high=2):
        self.first_winter = Dataset(first_winter_name)
        self.second_winter = Dataset(second_winter_name)
        self.min_pres = min_pres
        self.max_pres = max_pres
        self.int_pres_low = int_pres_low
        self.int_pres_high = int_pres_high
        self.interpol_pres_low = self.first_winter['pfull'][
                                        self.min_pres].data.item()
        self.interpol_pres_high = self.first_winter['pfull'][
                                        self.max_pres].data.item()
        self.interpol_lat_low = self.first_winter['lat']

    def get_useful_part(self, variable):
        first_winter_part = self.first_winter[variable][270:]
        second_winter_part = self.second_winter[variable][:120]
        winter = np.concatenate((first_winter_part, second_winter_part),
                                axis=0)
        winter = winter[:, self.min_pres:self.max_pres, :, :]
        winter = np.mean(winter, axis=3)
        winter = winter[:, self.int_pres_low:self.int_pres_high, :]
        return winter

    def get_uwind(self, lat):
        wind = self.get_useful_part('ucomp')
        if lat == 60:
            idx = 53
        elif lat == 65:
            idx = 55
        else:
            raise ValueError('Invalid option for the current definitions')

        wind = wind[:, :, idx]
        wind_interp = np.zeros(shape=wind.shape[0])
        for i in range(len(wind_interp)):
            x = np.array([self.interpol_pres_low, self.interpol_pres_high])
            y = np.array([wind[i][0], wind[i][1]])
            f = interp1d(x, y)
            wind_interp[i] = f(10.0)

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
