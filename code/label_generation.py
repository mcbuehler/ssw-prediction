import numpy as np
from preprocessor import Preprocessor



def UnT(data):
    """ Given a data matrix of winter, checks SSW events that conform to U&T definition:

        Events occur when the zonal-mean zonal winds at 10 hPa and
        60°N fall below 0 m s –1 from Nov to Mar. Events that do not also
        have a meridional temperature gradient reversal (defined as the
        zonal-mean temperatures averaged from 80° to 90°N minus the
        temperatures averaged from 60° to 70°N) within ~10 days of the
        circulation reversal are excluded. Events must return to westerly
        (>0 m s –1 ) for at least 20 consecutive days between events. The
        winds must return to westerly for at least 10 consecutive days
        prior to 30 Apr (or an event is considered a final warming).

                        Parameters
                        ----------
                             data: np.array which contains timeseries for
                             [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
                             in the given order

                        Returns
                        -------
                             SSWs: list[int]
                                A mask of booleans which includes SSW dates for each winter day

        """
    def check_SSW(m_temp_gradient, ssw):
        return any(x > 0 for x in m_temp_gradient[ssw - 10: ssw + 10])

    # temp_80_90 - temp_60_70
    m_temp_gradient = data[2] - data[1]
    potential_SSWs = SSWs_wind_reversal(data, 3)[1]

    SSWs = [ssw for ssw in potential_SSWs if check_SSW(m_temp_gradient, ssw)]
    SSWmask = np.zeros(data[1].shape, bool)

    SSWmask[SSWs] = True

    return SSWmask


def CP07(data):
    """ Given a data matrix of winter, checks SSW events that conform to CP07 definition:


                     Parameters
                        ----------
                             data: np.array which contains timeseries for
                             [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
                             in the given order

                        Returns
                        -------
                             SSWs: list[int]
                                A mask of booleans which includes SSW dates for each winter day
    """
    SSWs = SSWs_wind_reversal(data, 3)
    SSWmask = np.zeros(data[1].shape, bool)

    SSWmask[SSWs] = True
    return SSWmask


def U65(data):
    """Given a data matrix of winter, checks SSW events that conform to U65 definition:




                    Parameters
                    ----------
                         data: np.array which contains timeseries for
                         [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
                         in the given order

                    Returns
                    -------
                         SSWs: list[int]
                            A mask of booleans which includes SSW dates for each winter day
    """

    SSWs = SSWs_wind_reversal(data, 4)
    SSWmask = np.zeros(data[1].shape, bool)

    SSWmask[SSWs] = True
    return SSWmask


def SSWs_wind_reversal(data, data_index):
    """Given a data matrix of winter, checks whether a wind reversal that might be an SSW happened or not.

            Parameters
            ----------
                 data: np.array which contains timeseries for
                         [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
                         in the given order

                 data_index: index of which array is going to be used (i.e. wind_60, wind_65)

            Returns
            -------
                 SSWs: list
                    Indices of days when SSW events happen.

            """

    # an array of indices of the SSW events
    SSWs = []

    # temporary variable for keeping the potential SSW event of interest
    candidate_index = None

    # Number of consecutive days when the winds are westerly
    streak = 0

    for i, wind in enumerate(data[data_index, :]):

        # If wind is westerly, increase the streak
        if wind >= 0:
            streak += 1

            # If the winds return to westerly for at least 10 consecutive days
            # prior to 30 Apr and events happen from Nov to Mar
            if streak >= 10 and candidate_index is not None \
                    and candidate_index < 180:
                SSWs.append(candidate_index)
                candidate_index = None

        # If wind is reversed.
        else:

            # If we are not in the middle of an SSW event and
            # If this is the first time that the wind is reversed or we have
            # not experienced any wind reversals for 20 consecutive days,
            # start the explore this reversal as an SSW event.
            if candidate_index is None and (len(SSWs) == 0 or streak >= 20):
                candidate_index = i

            # reset streak
            streak = 0

    return SSWs


# TODO: Does not work, try to understand how it works for geopotential
def zpol_with_temp(data):
    # January February March
    jfm_timeseries = data[0][90:180]
    jfm_mean = np.mean(jfm_timeseries)
    jfm_std = np.std(jfm_timeseries)

    standardized_winter = (data[0] - jfm_mean) / jfm_std
    SSWs = [i for i, val in enumerate(standardized_winter) if val > 3]
    return len(SSWs) != 0, SSWs


definitions = {
    "U&T": UnT,
    "CP07": CP07,
    "U65": U65,
    "ZPOL_temp": zpol_with_temp
}


def create_labels(data, definition):
    if definition not in definitions:
        raise Exception(
            "Definition {} does not exist in available definitions: {}"
                .format(definition, list(definitions.keys())))

    f = definitions[definition]
    return [f(xi) for xi in data]


def get_available_defitinions():
    return list(definitions.keys())


if __name__ == '__main__':
    preprocess = Preprocessor('../data/atmos_daily_1.nc', '../data/atmos_daily_2.nc')
    wind_60 = preprocess.get_uwind(60)
    wind_65 = preprocess.get_uwind(65)
    temp = preprocess.get_polar_temp([(60, 70), (80, 90), (60, 90)])
    data = preprocess.construct_data_point(wind_60, wind_65, temp)

    # Create a dummmy array consisting of years
    dat = np.array([[data.temp_60_90, data.temp_60_70, data.temp_80_90, data.wind_60, data.wind_65],
                    [data.temp_60_90, data.temp_60_70, data.temp_80_90, data.wind_60, data.wind_65],
                    [data.temp_60_90, data.temp_60_70, data.temp_80_90, data.wind_60, data.wind_65],
                    [data.temp_60_90, data.temp_60_70, data.temp_80_90, data.wind_60, data.wind_65]])

    labels = create_labels(dat, "CP07")

    print(labels)
