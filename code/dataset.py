class Data:
    """The format of a data point after preprocessing has been done

    Attributes
    ----------
    temp_60_90: numpy array
        An array that has the polar cap temperatures for every day from
        latitude 60 to latitude 90.
    temp_60_70: numpy array
        An array that has the polar cap temperatures for every day from
        latitude 60 to latitude 70.
    temp_70_80: numpy array
        An array that has the polar cap temperatures for every day from
        latitude 70 to latitude 80.
    wind_60: numpy array
        An array that contains the U component of the wind at 60 latitude.
    wind_65: numpy array
        An array that contains the U component of the wind at 60 latitude.
    """
    temp_60_90 = None
    temp_60_70 = None
    temp70_80 = None
    wind_60 = None
    wind_65 = None
