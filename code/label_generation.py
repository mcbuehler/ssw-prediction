import h5py
import numpy as np
import os


def check_SSW(m_temp_gradient, ssw):
    return any(x > 0 for x in m_temp_gradient[ssw - 10: ssw + 10])


def UnT(data):
    """ Given a data matrix of winter, checks SSW events that conform
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
        data: np.array
            data which contains timeseries for
            [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
            in the given order


    Returns
    -------
         SSWs: list[int]
            A mask of booleans which includes SSW dates for each winter day

    """

    # temp_80_90 - temp_60_70
    m_temp_gradient = data[2] - data[1]
    potential_SSWs = SSWs_wind_reversal(data, 3)

    SSWs = [ssw for ssw in potential_SSWs if check_SSW(m_temp_gradient, ssw)]
    SSWmask = np.zeros(data[1].shape, bool)

    SSWmask[SSWs] = True

    return SSWmask


def CP07(data):
    """ Given a data matrix of winter, checks SSW events that conform
    to CP07 definition:

    Events occur when the zonal-mean zonal winds at 10 hPa and 60°N
    fall below 0 m s–1 from Nov to Mar. Events must return to westerly
    (>0 m s–1) for at least 20 consecutive days between events. The
    winds must return to westerly for at least 10 consecutive days prior
    to 30 Apr (or an event is considered a final warming).

    Parameters
    ----------
         data: np.array
            data which contains timeseries for
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
    """Given a data matrix of winter, checks SSW events that conform
    to U65 definition:

    Identical to CP07, except using zonal-mean zonal wind at 65°N.


    Parameters
    ----------
        data: np.array
        data which contains timeseries for
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
    """Given a data matrix of winter, checks whether a wind reversal
    that might be an SSW happened or not.

    Parameters
    ----------
         data: data: np.array
                data which contains timeseries for
                [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
                in the given order

         data_index: index of which array is going to be used
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


# TODO: Fix there is a problem with this defn.
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
        data: np.array
        data which contains timeseries for
        [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
        in the given order


    Returns
    -------
         SSWs: list[int]
            A mask of booleans which includes SSW dates for each winter day
    """

    # January February March timeseries
    jfm_timeseries = data[0][90:180]
    jfm_mean = np.mean(jfm_timeseries)
    jfm_std = np.std(jfm_timeseries)

    # standardization of winter time-series using JFM mean
    standardized_winter = (data[0] - jfm_mean) / jfm_std
    potentialSSWs = [i for i, val in enumerate(standardized_winter) if val > 3]
    SSWs = []

    # If potential SSW is in the first 3 months of winter and there
    # are at least 2 months between SSWs
    cur = None
    for SSW in potentialSSWs:
        if (cur is None or SSW - 60 > cur) and SSW < 180:
            cur = SSW
            SSWs.append(SSW)

    SSWmask = np.zeros(data[0].shape, bool)

    SSWmask[SSWs] = True
    return SSWmask


definitions = {
    "U&T": UnT,
    "CP07": CP07,
    "U65": U65,
    "ZPOL_temp": zpol_with_temp
}


def create_labels(data, definition):
    """Given a data matrix of winter and definition, checks whether an SSW happened
    that conforms to the given definition.

    Parameters
    ----------
         data: np.array
            data which contains timeseries for
            [temp_60_90, temp_60_70, temp_80_90, wind_60, wind_65]
            in the given order

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
    return [f(xi) for xi in data]


def get_available_definitions():
    return list(definitions.keys())


def label_dataset(path_in, path_out):
    print("Labelling data from {}".format(path_in))
    # Load data from h5 file
    f = h5py.File(path_in, 'r')

    # Get group names and dictionary names
    data_fields = \
        ['temp_60_90', 'temp_60_70', 'temp_80_90', 'wind_60', 'wind_65']
    keys = list(f.keys())

    print("Processing data...")
    # Get data and label it
    dat = np.array(
        [[f[key][data_field] for data_field in data_fields] for key in keys])

    print("Creating labels...")
    labels = [create_labels(dat, definition) for definition in
              get_available_definitions()]

    print("Writing labelled outputs...")
    # Persist the labeled data as a new h5 file
    f2 = h5py.File(path_out, "w")

    for i, key in enumerate(keys):
        g = f2.create_group(key)

        for var in data_fields:
            g.create_dataset(var, data=f[key][var], dtype=np.double)
        for j, label in enumerate(get_available_definitions()):
            g.create_dataset(label, data=labels[j][i], dtype=np.bool)

    f2.close()
    print("Written to {}".format(path_out))


if __name__ == '__main__':
    path_preprocessed = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT")
    path_in = os.path.join(path_preprocessed, "data_preprocessed.h5")
    path_out = os.path.join(path_preprocessed, "data_labeled.h5")
    label_dataset(path_in, path_out)
