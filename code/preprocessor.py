import numpy as np
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from bisect import bisect_right
from dataset import Data


class Preprocessor:
    """A class that takes two years as unprocessed model data and returns the
    variables of interest for a whole winter.

    Attributes
    ----------
    USEFUL_PRES: int
        The pressure of interest.
    START_DAY: int
        The start day of the winter in the first year.
    END_DAY: int
        The end day of the winter in the second year.
    TEMP_LAT: list
        The latitudes of interest that we have to interpolate in.
    """
    USEFUL_PRES = 10
    START_DAY = 270
    END_DAY = 120
    TEMP_LAT = [60, 70, 80]

    def __init__(self, first_winter_name, second_winter_name):
        """The constructor of the Preprocessor class

        Parameters
        ----------
            first_winter_name: string
                The path of the nc file of the first winter.
            second_winter_name: string
                The path of the nc file of the second winter.

        """

        self.first_winter = Dataset(first_winter_name)
        self.second_winter = Dataset(second_winter_name)

    def get_pres_interpol(self, search_variable, search_value, low_part=None):
        """Finds in an array the indices of the values that have to be
        interpolated using binary search.

        Parameters
        ----------
            search_variable: string
                The variable that we have to search in (e.g. latitude).
            search_value: int
                The value of the variable that needs to be searched.
            low_part: int, optional
                The value where you have to truncate the matrix from below in
                order to do the binary search

        Returns
        -------
             idx_before: int
                The index of the array before the value of interest.
             idx_after: int
                The index of the array after the value of interest.

        """
        if low_part is not None:
            var_to_search = self.first_winter[search_variable][low_part:].data
        else:
            var_to_search = self.first_winter[search_variable][:].data
        try:
            idx_after = bisect_right(var_to_search, search_value)
            idx_before = idx_after - 1
        except IndexError:
            raise IndexError("Cannot interpolate in the given model")

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
                An array of (210, 14) that has the temperature of the latitudes
                together with the interpolated values at the requested levels.
             latitudes list
                A list that contains the latitude levels and corresponds to the
                second dimension of the temp_interp numpy array.


        """

        temp = self.get_useful_part('temp')
        # remove the latitudes that we don't need
        idx_low_lat, _ = self.get_pres_interpol('lat', lower_lat)
        # keep the latitudes (we will need them for the calculation of the
        # polar cap mean)
        latitudes = list(self.first_winter['lat'][idx_low_lat:])
        temp = temp[:, :, idx_low_lat:]
        # keep only the pressure levels before and after 10 hPa
        idx_low_pres, idx_high_pres = self.get_pres_interpol('pfull',
                                                             self.USEFUL_PRES)
        temp_interp = np.zeros(shape=(temp.shape[0], temp.shape[2]))
        # interpolate though the pressure levels
        for i in range(len(temp_interp)):
            for j in range(len(temp_interp[i])):
                x = self.first_winter.variables[
                    'pfull'][idx_low_pres:idx_high_pres + 1].data
                y = temp[i, :, j]
                f = interp1d(x, y)
                temp_interp[i, j] = f(self.USEFUL_PRES)

        temp_interp_more = np.zeros(shape=(temp_interp.shape[0],
                                    temp_interp.shape[1] + len(self.TEMP_LAT))
                                    )
        start_ind = []
        level_info = []
        # find the right values for each latitude level
        for i, level in enumerate(self.TEMP_LAT):
            # find where you have to interpolate in the whole latitude matrix
            idx_low_level, idx_high_level = self.get_pres_interpol('lat',
                                                                   level)
            # find where you have to interpolate in the reduced latitude
            # matrix
            idx_lat_before, idx_lat_after = self.get_pres_interpol(
                                             'lat', level,
                                             low_part=idx_low_lat
                                             )
            start_ind.append(idx_lat_before + 1)
            x = self.first_winter.variables[
                    'lat'][idx_low_level: idx_high_level + 1].data

            latitudes.insert(idx_lat_before + 1 + i, level)
            level_info.append((idx_low_level, idx_high_level, idx_lat_before,
                               idx_lat_after, x))

        # do the interpolation for the latitude levels
        for i in range(len(temp_interp)):
            temp_var = temp_interp[i]
            for j in range(len(self.TEMP_LAT)):
                y = temp_interp[i][level_info[j][2]: level_info[j][3] + 1]
                f = interp1d(level_info[j][4], y)
                temp_var = np.insert(temp_var, start_ind[j] + j,
                                     f(self.TEMP_LAT[j]))
            temp_interp_more[i] = temp_var

        return temp_interp_more, latitudes

    def get_polar_temp(self, limits):
        """Computes the polar cap temperature average based on the limits
        provided.

        Parameters
        ----------
            limits: list of tuples
                A list that contains the tuples of the the limits in the format
                (lower_limit, upper_limit).

        Returns
        -------
             average_temp: numpy array
                An array that has the polar cap temperatures for every day
                with dimensions (210, 3).


        """
        # feed the minimum latitude in the preprocess_temp method
        temp, latitudes = self.preprocess_temp(
                lower_lat=min(limits, key=lambda t: t[0])[0])
        # computed the weights for the weighted average
        weights = np.cos(np.deg2rad(latitudes))

        # find the indices of the latitudes for the temperatures of interest
        limits_idx = []
        for limit in limits:
            lower_lat_idx = bisect_right(latitudes, limit[0]) - 1
            try:
                higher_lat_idx = bisect_right(latitudes, limit[1])
            except IndexError:
                higher_lat_idx = None
            limits_idx.append((lower_lat_idx, higher_lat_idx))

        # compute the indices of the temperatures of interest
        average_temp = np.zeros(shape=(temp.shape[0], len(limits)))
        for i in range(len(temp)):
            for j in range(len(limits)):
                average_temp[i, j] = np.average(
                                                temp[i][limits_idx[j][0]:
                                                        limits_idx[j][1]],
                                                weights=weights[
                                                    limits_idx[j][0]:
                                                    limits_idx[j][1]
                                                    ]
                                                )

        return average_temp

    def construct_data_point(self, wind_60, wind_65, temp):
        """Gets the necessary data and produces a data point based on a Data class

        Parameters
        ----------
            wind_60: numpy array
                An array that contains the wind at 60 latitude for every day
                with dimensions (210, 1).
            wind_65: numpy array
                An array that contains the wind at 65 latitude for every day
                with dimensions (210, 1).
            temp: numpy array
                An array that has the polar cap temperatures for every day
                with dimensions (210, 3).

        Returns
        -------
             data: class instance
                An Data class instance with attributes 'temp_60_70',
                'temp_60_80', 'temp_60_90', 'wind_60', 'wind_65'.


        """
        data = Data()
        data.temp_60_70 = temp[:, 0]
        data.temp_80_90 = temp[:, 1]
        data.temp_60_90 = temp[:, 2]
        data.wind_60 = wind_60
        data.wind_65 = wind_65
        return data


if __name__ == '__main__':
    preprocess = Preprocessor('data/atmos_daily_1.nc', 'data/atmos_daily_2.nc')
    wind_60 = preprocess.get_uwind(60)
    wind_65 = preprocess.get_uwind(65)
    temp = preprocess.get_polar_temp([(60, 70), (80, 90), (60, 90)])
    data = preprocess.construct_data_point(wind_60, wind_65, temp)
    # print(data.__dict__)
