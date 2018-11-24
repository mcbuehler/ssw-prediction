import numpy as np
from dataset import Datapoint
from netCDF4 import Dataset


class PreprocessorReal:
    """
    A class to preprocess, takes two years as real data from JRA-55 data
    and returns the variables of interest for a whole winter.
    """

    # Pressure level of interest
    USEFUL_PRES = 10000

    # A cutoff which specifies how many days we should get from the end of the
    # first year. Because of leap years, the starting day of the winter
    # is not constant
    DAYS_BEFORE_START_DAY = 92

    # A cutoff which specifies how many days should we get from the beginning
    # of the second year.
    END_DAY = 120

    def __init__(self, first_winter_u_name, second_winter_u_name,
                 first_winter_t_name, second_winter_t_name):
        """
        Initializer for PreprocessorReal class.
        Since each input feature is located at different files, we need to
        specify a different file name for each feature and year.

        :param first_winter_u_name: The path of nc of the
        first winter's wind (u component)
        :param second_winter_u_name: The path of nc file of the
        second winter's wind (u component)
        :param first_winter_t_name: The path of nc file of the
        first winter's temperature
        :param second_winter_t_name: The path of nc file of the
        second winter's temperature
        """

        self.first_winter = {"u": Dataset(first_winter_u_name),
                             "t": Dataset(first_winter_t_name)}

        # Always get the index of 92 days before the last day.
        # This index needs to be dynamic because of the leap years
        n_days = len(self.first_winter["u"]["time"][:].data)
        self.start_day = n_days - self.DAYS_BEFORE_START_DAY

        self.second_winter = {"u": Dataset(second_winter_u_name),
                              "t": Dataset(second_winter_t_name)}

    def get_uwind(self, lat):
        """
        Returns the time series of u component of the wind at a specified
        latitude of interest.

        :param lat: Latitude of interest
        :return: Time series for u wind at the latitude of interest with
        shape (212,)
        """

        winter = self.get_useful_part("u")

        # Get searched latitude
        ind, _ = self.get_dimension_index("lat", lat)
        winter = np.take(winter, ind, axis=1)

        return winter

    def get_polar_avg_temp(self, lat_min, lat_max):
        """
        Returns the time series of temperature  specified latitude of
        interest.

        :param lat_min: Latitude of interest (south of lat_min)
        :param lat_max: Latitude of interest (north of lat_max)
        :return: Time series for temperature between latitudes of interest
        (cosine averaged) with shape (212,)
        """
        winter = self.get_useful_part("t")

        # Get searched latitudes' indexes
        ind_min, latitudes = self.get_dimension_index("lat", lat_min)
        ind_max, _ = self.get_dimension_index("lat", lat_max)

        # Calculate weights for each latitude
        weights = np.cos(np.deg2rad(latitudes[ind_max:ind_min]))

        # Cosine weighted averaging of temperatures
        winter = winter[:, ind_max:ind_min]
        winter = np.average(winter, axis=1, weights=weights)

        return winter

    def get_dimension_index(self, dimension, value):
        """
        For a given dimension, searches for the index where it is equal to a
        certain value. (example: index of the wind at latitude 60, index of
        pressure level when it is equal to 10000)

        :param dimension: dimension of interest (example: latitudes)
        :param value: value of interest (example: 60)
        :return: A tuple that contains the index of the value of
        interest and values for the dimension. (index, available_levels)
        """
        # Get an arbitrary dataset to look into the dimensions
        dataset = self.first_winter['u']

        # Get available levels for the dimension of interest
        available_levels = list(dataset[dimension][:].data)

        # Get the index where it is equal to the value of interest
        index = np.where(np.isclose(available_levels, value))[0]
        if len(index) == 0:
            raise LookupError("{} could not be found at {} level.".
                              format(dimension, value))

        return np.asscalar(index), available_levels

    def get_useful_part(self, variable):
        """
        For a given variable of interest, preprocesses the dataset to derive
        winterly time series for each latitude. Longitudes are averaged, only
        the pressure in the self.USEFUL_PRES level is extracted.

        :param variable: Variable of interest (example: "u", "t")
        :return: Winterly time series (212, 145) corresponding to
        (days, latitudes).
        """

        # Get the winter part from the two years and concatenate them
        first_winter = self.first_winter[variable][variable][self.start_day:]
        second_winter = self.second_winter[variable][variable][:self.END_DAY]

        winter = np.concatenate((first_winter, second_winter),
                                axis=0)

        # Get proper pressure level index
        ind, _ = self.get_dimension_index("lev", self.USEFUL_PRES)
        # Filter the winter such that it only includes
        winter = np.take(winter, ind, axis=1)

        # Average through longitudes
        winter = np.mean(winter, axis=2)

        return winter

    def construct_data_point(self, wind_60, wind_65, temp_60_70, temp_80_90,
                             temp_60_90):
        """
        Gets the necessary data and produces a data point based on a Data class

        :param wind_60: An array that contains the wind at 60 latitude for
        every day with dimensions (212, 1).
        :param wind_65: An array that contains the wind at 65 latitude for
        every day with dimensions (212, 1).
        :param temp_60_70: An array that contains the temperature cosine
        averaged between 60-70 latitude for every day with dimensions (212, 1)
        :param temp_80_90: An array that contains the temperature cosine
        averaged between 80-90 latitude for every day with dimensions (212, 1)
        :param temp_60_90: An array that contains the temperature cosine
        averaged between 60-90 latitude for every day with dimensions (212, 1)
        :return: A Datapoint class instance with attributes 'temp_60_70',
                'temp_60_80', 'temp_60_90', 'wind_60', 'wind_65'.
        """
        data_dict = dict(
            temp_60_70=temp_60_70,
            temp_80_90=temp_80_90,
            temp_60_90=temp_60_90,
            wind_60=wind_60,
            wind_65=wind_65)
        data_point = Datapoint(**data_dict)
        return data_point


class RealDataPointFactory:
    """
    Factory class for creating datapoints
    """

    @staticmethod
    def create(first_winter_u_name, second_winter_u_name,
               first_winter_t_name, second_winter_t_name):
        """

        :param first_winter_u_name: The path of nc of the
        first winter's wind (u component)
        :param second_winter_u_name: The path of nc of the
        second winter's wind (u component)
        :param first_winter_t_name: The path of nc file of the
        first winter's temperature
        :param second_winter_t_name: The path of nc file of the
        second winter's temperature
        :return: Datapoint
        """
        preprocess = PreprocessorReal(first_winter_u_name,
                                      second_winter_u_name,
                                      first_winter_t_name,
                                      second_winter_t_name)

        wind_60 = preprocess.get_uwind(60)
        wind_65 = preprocess.get_uwind(65)
        temp_60_70 = preprocess.get_polar_avg_temp(60, 70)
        temp_80_90 = preprocess.get_polar_avg_temp(80, 90)
        temp_60_90 = preprocess.get_polar_avg_temp(60, 90)

        data = preprocess.construct_data_point(wind_60, wind_65, temp_60_70,
                                               temp_80_90, temp_60_90)
        return data


if __name__ == '__main__':
    preprocess = PreprocessorReal(
        "../../data/u-jra55-125-daymean-1959.nc",
        "../../data/u-jra55-125-daymean-1960.nc",
        "../../data/t-jra55-125-daymean-1959.nc",
        "../../data/t-jra55-125-daymean-1960.nc")

    wind_60 = preprocess.get_uwind(60)
    wind_65 = preprocess.get_uwind(65)
    temp_60_70 = preprocess.get_polar_avg_temp(60, 70)
    temp_80_90 = preprocess.get_polar_avg_temp(80, 90)
    temp_60_90 = preprocess.get_polar_avg_temp(60, 90)
    data = preprocess.construct_data_point(wind_60, wind_65, temp_60_70,
                                           temp_80_90, temp_60_90)

    print(data)
