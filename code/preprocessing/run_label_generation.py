import os
import h5py
import numpy as np
from dataset import DatapointKey
from data_manager import DataManager


def check_SSW(m_temp_gradient, ssw):
    """Checks whether a potential SSW also causes a meridional temperature
    gradient reversal (defined as the zonal-mean temperatures averaged
    from 80° to 90°N minus the temperatures averaged from 60° to 70°N)

    :param m_temp_gradient: Array which contains meridional temperature
    gradient through a winter.
    :param ssw: Potential indexes for SSW events
    :return: Indices of SSW events that conform to the U&T definition
    """
    return any(x > 0 for x in m_temp_gradient[ssw - 10: ssw + 10])


def UnT(xi):
    """Given a data matrix of winter, checks SSW events that conform
    to U&T definition:

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
        xi:dict
         data which contains time series for [temp_60_90, temp_60_70,
         temp_80_90, wind_60, wind_65] as a dictionary for one year


    Returns
    -------
         SSWs: list[int]
            A mask of booleans which includes SSW dates for each winter day

    """

    # temp_80_90 - temp_60_70
    m_temp_gradient = xi[DatapointKey.TEMP_80_90] - xi[DatapointKey.TEMP_60_70]
    potential_SSWs = SSWs_wind_reversal(xi, DatapointKey.WIND_60)

    SSWs = [ssw for ssw in potential_SSWs if
            check_SSW(m_temp_gradient, ssw)]

    SSWmask = np.zeros(xi[DatapointKey.WIND_60].shape, bool)

    SSWmask[SSWs] = True

    return SSWmask


def CP07(xi):
    """Given a data matrix of winter, checks SSW events that conform
    to CP07 definition:

    Events occur when the zonal-mean zonal winds at 10 hPa and 60°N
    fall below 0 m s–1 from Nov to Mar. Events must return to westerly
    (>0 m s–1) for at least 20 consecutive days between events. The
    winds must return to westerly for at least 10 consecutive days prior
    to 30 Apr (or an event is considered a final warming).

    Parameters
    ----------
         xi:dict
         data which contains time series for [temp_60_90, temp_60_70,
         temp_80_90, wind_60, wind_65] as a dictionary for one year


    Returns
    -------
         SSWs: list[int]
            A mask of booleans which includes SSW dates for each winter day
    """

    SSWs = SSWs_wind_reversal(xi, DatapointKey.WIND_60)
    SSWmask = np.zeros(xi[DatapointKey.WIND_60].shape, bool)

    SSWmask[SSWs] = True
    return SSWmask


def U65(xi):
    """Given a data matrix of winter, checks SSW events that conform
    to U65 definition:

    Identical to CP07, except using zonal-mean zonal wind at 65°N.


    Parameters
    ----------
        xi:dict
         data which contains time series for [temp_60_90, temp_60_70,
         temp_80_90, wind_60, wind_65] as a dictionary for one year


    Returns
    -------
         SSWs: list[int]
            A mask of booleans which includes SSW dates for each winter day
    """

    SSWs = SSWs_wind_reversal(xi, DatapointKey.WIND_65)
    SSWmask = np.zeros(xi[DatapointKey.WIND_65].shape, bool)

    SSWmask[SSWs] = True
    return SSWmask


def SSWs_wind_reversal(xi, datatype_name):
    """Given a data matrix of winter, checks whether a wind reversal
    that might be an SSW happened or not.

    Parameters
    ----------
         xi:dict
         data which contains time series for [temp_60_90, temp_60_70,
         temp_80_90, wind_60, wind_65] as a dictionary for one year

         datatype_name: name of which numpy array is going to be used
         (i.e. wind_60, wind_65)

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

    for i, wind in enumerate(xi[datatype_name][:]):

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


def zpol_with_temp(data):
    """Given a data matrix of winter, checks SSW events that conform
    to ZPOL definition, but uses temperature instead of geopotential:

    Anomalies of zonal-mean geopotential heights at 10 hPa are found
    following Gerber et al. (2010). The polar cap anomalies are found
    by averaging (cosine weighted) anomalies from 60° to 90°N. This
    (year-round) time series is standardized about the JFM mean (as in
    Thompson et al. 2002). Events occur when the time series exceed
    plus three standard deviations. An event that occurs within 60
    days after another is excluded.


    Parameters
    ----------
        data:dict
         data which contains time series for [temp_60_90, temp_60_70,
         temp_80_90, wind_60, wind_65] as a dictionary for each year
         in the dataset.


    Returns
    -------
         SSWs: list[int]
            A mask of booleans which includes SSW dates for each winter day
    """

    def zpol_with_temp_single(xi, jfm_mean, jfm_std):

        # Standardization of winter time-series using JFM mean
        standardized_winter = (xi[DatapointKey.TEMP_60_90] - jfm_mean) / jfm_std

        # Label as an SSW event if the temperature is 3 stdev more
        # than JFM mean
        potentialSSWs = [i for i, val in enumerate(standardized_winter)
                         if val > 3]
        SSWs = []

        # If potential SSW is in the first 3 months of winter and there
        # are at least 2 months between SSWs
        cur = None
        for SSW in potentialSSWs:
            if (cur is None or SSW - 60 > cur) and SSW < 180:
                cur = SSW
                SSWs.append(SSW)

        SSWmask = np.zeros(xi[DatapointKey.TEMP_60_90].shape, bool)

        SSWmask[SSWs] = True
        return SSWmask

    # For each year, extract the time series for temp_60_90 between days
    # 90 - 180 -> January, February, March
    jfm_timeseries = data[DatapointKey.TEMP_60_90][:, 90:180]

    # Take mean for each winter (JFM time series of temp_60_90)
    jfm_mean = np.mean(jfm_timeseries, axis=1)

    # Get number of years in the data
    number_of_years = len(jfm_mean)

    # Take standard deviation of these means to derive JFM stdev
    jfm_std = np.std(jfm_mean)
    jfm_mean = np.mean(jfm_mean)

    # Create a list of dictionaries which contain each data field
    # for a year to iterate
    winters = [{data_field: data[data_field][i, :] for data_field in data.keys()}
               for i in range(number_of_years)]

    return [zpol_with_temp_single(xi, jfm_mean, jfm_std) for xi in winters]


definitions = {
    DatapointKey.UT: UnT,
    DatapointKey.CP07: CP07,
    DatapointKey.U65: U65,
    DatapointKey.ZPOL: zpol_with_temp
}


def create_labels(data, definition):
    """Given a data matrix of winter and definition,
    checks whether an SSW happened
    that conforms to the given definition.

    Parameters
    ----------
         data: dict
            data which contains time series for
            [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
            as a dictionary

         definition: str
                definition type (i.e. CP07)

    Returns
    -------
         SSWs: list[int]
                A mask of booleans which includes SSW dates for each winter day

    """
    if definition not in definitions:
        raise Exception(
            "Definition {} does not exist in available definitions: {}"
            .format(definition, list(definitions.keys())))

    f = definitions[definition]

    if definition == DatapointKey.ZPOL:
        return f(data)
    else:
        # Get number of years in the data set
        number_of_years = len(data[DatapointKey.WIND_60])

        # Create a list of dictionaries which contain each data field
        # for a year to iterate
        winters = [{data_field: data[data_field][i, :] for data_field in data.keys()}
                   for i in range(number_of_years)]

        return [f(xi) for xi in winters]



def get_available_definitions():
    return list(definitions.keys())


def label_dataset(path_in, path_out):
    print("Labelling data from {}".format(path_in))
    # Load data from h5 file
    f = h5py.File(path_in, "r")

    # To persist the labeled data as a new h5 file
    f2 = h5py.File(path_out, "a")

    # Create a datamanager for input data.
    data_manager = DataManager(path_in)

    # Get group names and dictionary names
    data_fields = [DatapointKey.TEMP_60_90,
                   DatapointKey.TEMP_60_70,
                   DatapointKey.TEMP_80_90,
                   DatapointKey.WIND_60,
                   DatapointKey.WIND_65]

    keys = list(set(f.keys()) - set(f2.keys()))

    if len(keys) > 0:
        print("Processing data...")

        dat = {data_field: data_manager.get_data_for_variable(data_field)
               for data_field in data_fields}

        print("Creating labels...")
        labels = [create_labels(dat, definition) for definition in
                  get_available_definitions()]

        print("Writing labelled outputs...")

        for i, key in enumerate(keys):
            g = f2.create_group(key)

            for var in data_fields:
                g.create_dataset(var, data=f[key][var], dtype=np.double)
            for j, label in enumerate(get_available_definitions()):
                g.create_dataset(label, data=labels[j][i], dtype=np.bool)

        print("Written to {}".format(path_out))

    else:
        print("Labeled data up-to-date! No keys to add.")

    f2.close()


if __name__ == '__main__':
    path_preprocessed = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT")
    path_in = os.path.join(path_preprocessed, "data_preprocessed.h5")
    path_out = os.path.join(path_preprocessed, "data_labeled.h5")
    label_dataset(path_in, path_out)
