from collections import Iterable

import h5py

from dataset import Datapoint
from file_system_utils import get_files_with_prefix, get_data_file_paths
from preprocessor import DataPointFactory
import os
import pandas as pd
from joblib import Parallel, delayed

# Path and file name configuration
#     Production
PATH_BASE_INPUT = "/mnt/ds3lab-scratch/dslab2018/bernatj/model"
PATH_OUT_BASE = "/mnt/ds3lab-scratch/dslab2018_climate"
#     Development
# PATH_OUT_BASE = "data/preprocessing_out/"
# PATH_BASE_INPUT = "data"

#
PATH_OUT_HDF5 = os.path.join(PATH_OUT_BASE, "data_preprocessed.h5")
FOLDER_PREFIXES = ["SSW_clim_sst_", "fixed_sst_"]
DATA_FILE_NAME = "atmos_daily.nc"

# Job Configuration
LIMIT = 10  # Number of winters to process
N_JOBS = 12  # Number of cores to use
VERBOSE = 20  # verbosity level
# If True clears all previous processed data
# If False previously processed winters will be kept and not replaced
CLEAR_PREVIOUS = False


def get_data_paths(base_path, folder_prefixes, data_file_name,
                   limit=-1) -> list:
    """
    Returns paths to the files that should be processed
    :param base_path: folder where the single years are stored,
    e.g. "/mnt/ds3lab-scratch/dslab2018/bernatj/model"
    :param folder_prefixes: list with prefixes of folders
    that should be searched for data files, e.g. ["SSW_clim_sst_pert_"]
    :param data_file_name: str e.g. "atmos_daily.nc"
    :param limit: int number of winters to process. Use -1 to process all.
    :return: list of paths
    """
    # All input folders that start with one of the prefixes
    input_folders = []
    for prefix in folder_prefixes:
        input_folders += get_files_with_prefix(base_path, prefix,
                                               keep_path=True)

    # All paths to data files
    paths = list()
    for f in input_folders:
        data_file_paths = get_data_file_paths(f, data_file_name)
        paths += data_file_paths

    # We might want to keep previously processed datapoints
    if not CLEAR_PREVIOUS:
        # We want to keep the data we have already processed in a previous run
        preprocessed_winters = load_preprocessed_winters_identifiers(
            PATH_OUT_HDF5)
        paths = remove_datapoints(paths, preprocessed_winters)

    # Limit the number of processed winters,
    # e.g. for development or computational reasons
    if limit > 0:
        paths = paths[:limit]
    return paths


def remove_datapoints(paths: list, preprocessed_winters: list) -> list:
    """
    Removes preprocessed winters from
    :param paths: list of paths to data files
    :param preprocessed_winters: list of datapoint identifiers
    that should be removed
    :return: list of paths without preprocessed winters
    """
    already_processed_paths = list()
    for path in paths:
        folder, year = os.path.split(path)[-2].split(os.sep)[-2:]
        datapoint_identifier = Datapoint.create_datapoint_identifier(folder,
                                                                     year)
        if datapoint_identifier in preprocessed_winters:
            already_processed_paths.append(path)

    paths = [path for path in paths if path not in already_processed_paths]
    return paths


def process_single_year(path1: str, path2: str) -> tuple:
    """
    Processes the year contained in the files for the given paths
    :param path1: path of starting winter
    :param path2: path of ending winter
    :return: identifier for that winter, datapoint for that winter
    """
    path_elements = path1.split(os.sep)
    identifier = Datapoint.create_datapoint_identifier(path_elements[-3],
                                                       path_elements[-2])
    print("Processing {}...".format(identifier))
    datapoint = DataPointFactory.create(path1, path2)
    return identifier, datapoint


def write_csv(path_h5file, base_path_out):
    """
    Copies datapoints from h5 file to N CSV files
    (N = number of data points in h5 file).
    Filename will be the datapoint identifier with .csv postfix.
    :param path_h5file:
    :param base_path_out:
    :return:
    """
    file = h5py.File(path_h5file, 'r')
    groups = file.keys()
    for group in groups:
        data = {key: file[group][key] for key in file[group].keys()}
        df = pd.DataFrame(data)
        path_out = os.path.join(base_path_out, group)
        df.to_csv(path_out + ".csv", sep=',')


def load_preprocessed_winters_identifiers(path_h5_file: str) -> list:
    """
    Loads h5 file and return list of identifiers
    :param path_h5_file:
    :return:
    """
    file = h5py.File(path_h5_file, 'r')
    identifiers = file.keys()
    return identifiers


def run_preprocessing(limit=-1):
    """
    Runs the preprocessing for the configuration given in the global variables.
    Make sure to set these variables beforehand:
    - PATH_BASE_INPUT
    - FOLDER_PREFIXES
    - DATA_FILE_NAME
    - N_JOBS
    - PATH_OUT_HDF5
    :param limit: max number of winters to process.
    Set to -1 to process all winters.
    :return: None
    """
    # load all the paths we want to process
    paths = get_data_paths(PATH_BASE_INPUT, FOLDER_PREFIXES, DATA_FILE_NAME,
                           limit)
    # Run the job for all years
    dataset = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(process_single_year)(paths[i], paths[i + 1]) for i in
        range(len(paths) - 1))
    # Convert list to dict
    # dict_keys are dataset identifiers: ['SSW_clim_sst_pert_2_year_2', ..]
    dataset = {elem[0]: elem[1] for elem in dataset}

    # All the variables we want to save
    variables = Datapoint.get_variables()
    # Output h5 file
    out_file = h5py.File(PATH_OUT_HDF5, 'w')
    for key, data in dataset.items():
        # Group is one data point, e.g. SSW_clim_sst_pert_2_year_2
        g = out_file.create_group(key)
        for variable in variables:
            # variable is a feature variable, e.g. wind_65
            g[variable] = getattr(data, variable)
    out_file.close()


def run():
    # Process all years and save them to h5 file
    run_preprocessing(LIMIT)
    # Write the contents of the hdf5 file to csv files
    write_csv(PATH_OUT_HDF5, PATH_OUT_BASE)


if __name__ == "__main__":
    run()
