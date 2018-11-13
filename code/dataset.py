import sys
from collections import Iterable
from enum import Enum

import numpy as np


class Datapoint:
    """The format of a data point after preprocessing has been done

    Attributes
    ----------
    temp_60_70: numpy array
        An array that has the polar cap temperatures for every day from
        latitude 60 to latitude 70.
    temp_80_90: numpy array
        An array that has the polar cap temperatures for every day from
        latitude 70 to latitude 80.
    temp_60_90: numpy array
        An array that has the polar cap temperatures for every day from
        latitude 60 to latitude 90.
    wind_60: numpy array
        An array that contains the U component of the wind at 60 latitude.
    wind_65: numpy array
        An array that contains the U component of the wind at 60 latitude.
    """
    temp_60_70 = None
    temp_80_90 = None
    temp_60_90 = None
    wind_60 = None
    wind_65 = None

    def __init__(self, temp_60_70, temp_80_90, temp_60_90, wind_60, wind_65):
        self.temp_60_70 = temp_60_70
        self.temp_80_90 = temp_80_90
        self.temp_60_90 = temp_60_90
        self.wind_60 = wind_60
        self.wind_65 = wind_65

    def to_np_array(self) -> np.array:
        """

        :return: np.array
        """
        variables = [self.temp_60_70, self.temp_80_90, self.temp_60_90,
                     self.wind_60, self.wind_65]
        if any(elem is None for elem in variables):
            print("We have None in our variables!", file=sys.stderr)
            return None

        return np.array(variables)

    @staticmethod
    def get_variables() -> Iterable:
        variables = ["temp_60_70", "temp_80_90", "temp_60_90", "wind_60",
                     "wind_65"]
        return variables

    @staticmethod
    def create_datapoint_identifier(simulation_name: str, subfolder: str,
                                    file: str) -> str:
        """
        Creates an identifier for a datapoint given its folder and year
        :param folder: folder name, e.g. SSW_clim_sst_pert_1
        :param year: year, e.g. year_81
        :return: identifier for a datapoint, e.g.
        SSW_clim_sst_pert_1_year_81_atmos_daily
        """
        return "{}_{}_{}".format(simulation_name, subfolder, file[:-3])


class DatapointKey:
    TEMP_60_70 = "temp_60_70"
    TEMP_80_90 = "temp_80_90"
    TEMP_60_90 = "temp_60_90"
    WIND_60 = "wind_60"
    WIND_65 = "wind_65"

    CP07 = "CP07"
    UT = "U&T"
    U65 = "U65"
    ZPOL = "ZPOL_temp"

