import h5py

from dataset import Datapoint
from file_system_utils import get_files, get_data_file_paths
from preprocessor import DataPointFactory
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import signal
import sys


def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


def check_environment_variables():
    required_environ_variables = ["DSLAB_CLIMATE_BASE_INPUT",
                                  "DSLAB_CLIMATE_BASE_OUTPUT"]
    missing_environ_variables = [var for var in required_environ_variables if
                                 os.getenv(var) is None]

    if len(missing_environ_variables) > 0:
        print(
            "Please set environment variables before running this script:")
        print(", ".join(missing_environ_variables))
        exit(0)


# Path and file name configuration
check_environment_variables()
PATH_INPUT_BASE = os.getenv("DSLAB_CLIMATE_BASE_INPUT")
PATH_OUTPUT_BASE = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT")

PATH_OUT_HDF5 = os.path.join(PATH_OUTPUT_BASE, "data_preprocessed.h5")
FOLDER_PREFIXES = ["SSW_clim_sst_", "fixed_sst_"]
SUBFOLDER_PREFIXES = ["year_", "daymean"]
DATA_FILE_NAME = "atmos_daily.nc"

# Job Configuration
LIMIT = int(os.getenv("DSLAB_LIMIT", -1))  # Number of winters to process
N_JOBS = int(os.getenv("DSLAB_N_JOBS", 12))  # Number of cores to use
print("N_JOBS: ", N_JOBS)
VERBOSE = 20  # verbosity level
# If True clears all previous processed data
# # If False previously processed winters will be kept and not replaced
# CLEAR_PREVIOUS = False
CLEAR_PREVIOUS = os.getenv("DSLAB_CLEAR_PREVIOUS", 0) == "1"
print("CLEAR PREVIOUS: ", CLEAR_PREVIOUS)


def remove_datapoints(paths: list, preprocessed_winters: list,
                      verbose: bool = False) -> list:
    """
    Removes preprocessed winters from
    :param paths: list of paths to data files
    :param preprocessed_winters: list of datapoint identifiers
    that should be removed
    :param verbose
    :return: list of paths without preprocessed winters
    """
    already_processed_paths = list()
    for path in paths:
        folder, year = os.path.split(path)[-2].split(os.sep)[-2:]
        datapoint_identifier = Datapoint.create_datapoint_identifier(folder,
                                                                     year)
        if datapoint_identifier in preprocessed_winters:
            already_processed_paths.append(path)

    paths_reduced = list(
        path for path in paths if path not in already_processed_paths)

    if verbose:
        print("Ommitting {} already processed data points"
              .format(len(paths) - len(paths_reduced)))
    return paths_reduced


def process_single_year(path1: str, path2: str,
                        verbose: bool = False,
                        ommit_identifiers: list = list()) -> tuple:
    """
    Processes the year contained in the files for the given paths
    :param path1: path of starting winter
    :param path2: path of ending winter
    :param verbose
    :return: identifier for that winter, datapoint for that winter
    """
    path_elements = path1.split(os.sep)
    identifier = Datapoint.create_datapoint_identifier(path_elements[-3],
                                                       path_elements[-2],
                                                       path_elements[-1])
    if identifier not in ommit_identifiers:
        if verbose:
            print("Processing {}... (paths: {} and {})".format(identifier,
                                                               path1, path2))
        datapoint = DataPointFactory.create(path1, path2)
        return identifier, datapoint
    return None, None


def write_csv(path_h5file: str, base_path_out: str) -> None:
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
    file = h5py.File(path_h5_file, 'a')
    identifiers = list(file.keys())
    file.close()
    return identifiers


def run_single_simulation(simulation_folder, out_file: h5py.File,
                          ommit_identifiers: list = list()) -> int:
    """
    Runs all winters in given simulation.
    :param simulation_folder: folder containing one subfolder
    containing simulation data for all years.
    :param out_file: output h5 file
    :param ommit_identifiers: list of identifiers that should
    not be processed again.
    :return: number of processed winters
    """

    # load all the paths we want to process
    paths = get_data_file_paths(
        os.path.join(PATH_INPUT_BASE, simulation_folder))
    print("Processing {} datasets for simulation data in {}...".format(
        len(paths), simulation_folder))

    # Run the job for all years
    dataset = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(process_single_year)(paths[i], paths[i + 1], True,
                                     ommit_identifiers) for i in
        range(len(paths) - 1))
    # Convert list to dict
    # dict_keys are dataset identifiers: ['SSW_clim_sst_pert_2_year_2', ..]
    dataset = {elem[0]: elem[1] for elem in dataset if elem[0] is not None}

    # All the variables we want to save
    variables = Datapoint.get_variables()

    for key, data in dataset.items():
        # Group is one data point, e.g. SSW_clim_sst_pert_2_year_2
        g = out_file.create_group(key)
        for variable in variables:
            # variable is a feature variable, e.g. wind_65
            np_data = getattr(data, variable)
            # Let's make sure we actually have data in that variable
            assert isinstance(np_data, np.ndarray) and np_data is not None
            g.create_dataset(variable, data=np_data, dtype=np.double)
    return len(dataset.keys())


def run_preprocessing(limit: int = -1) -> None:
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
    # Collect all simulation folders, whatever they might be named
    # The criterion is that they start with one of the prefixes
    # in FOLDER_PREFIXES
    simulation_folders = list()
    for folder_prefix in FOLDER_PREFIXES:
        simulation_folders += get_files(PATH_INPUT_BASE, prefix=folder_prefix,
                                        keep_path=True, order_numerical=True)
    print("Found the following simulation folders:")
    print("\n".join(simulation_folders))

    # Used for having an optional limit on the number of processed winters
    # We keep track of how many winters we processed in each simulation.
    n_years_processed = list()

    # For avoiding to process winters a second time.
    preprocessed_winters = load_preprocessed_winters_identifiers(
        PATH_OUT_HDF5)
    print("Found {} already preprocessed winters.".format(
        len(preprocessed_winters)))

    # Output h5 file
    out_file = h5py.File(PATH_OUT_HDF5, 'a')

    for simulation_folder in simulation_folders:
        n_years = run_single_simulation(simulation_folder, out_file,
                                        ommit_identifiers=preprocessed_winters)
        n_years_processed.append(n_years)
        if np.sum(n_years_processed) > limit > 0:
            # We don't want to keep processing further
            break

    print("Processed {} years in {} simulation runs.".format(
        np.sum(n_years_processed), len(n_years_processed)))

    out_file.close()


def run():
    # Signal listener to shut down process in case it is interrupted
    signal.signal(signal.SIGINT, signal_handler)

    # We remove the old h5 file
    if CLEAR_PREVIOUS and os.path.exists(PATH_OUT_HDF5):
        input(
            "Are you sure you want to delete {}? Press Ctrl+C to \
abort or enter to continue".format(PATH_OUT_HDF5))
        os.remove(PATH_OUT_HDF5)
        print("Removed {}".format(PATH_OUT_HDF5))
    # Process all years and save them to h5 file
    run_preprocessing(LIMIT)
    # Write the contents of the hdf5 file to csv files
    write_csv(PATH_OUT_HDF5, PATH_OUTPUT_BASE)


if __name__ == "__main__":
    run()
