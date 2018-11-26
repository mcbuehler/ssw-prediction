import os
from preprocessor_real import RealDataPointFactory
from dataset import Datapoint
import h5py
import numpy as np

# Constants:

# Environment variable to get input directory of nc files
INPUT_DIR_KEY = "DSLAB_CLIMATE_BASE_INPUT_REAL"

# Environment variable to get output directory to store h5 file
OUTPUT_DIR_KEY = "DSLAB_CLIMATE_BASE_OUTPUT_REAL"

# Years to preprocess
INTERVAL = (1958, 2016)

# Format to crate group names for h5 file
FILE_FORMAT = "/{}/{}-jra55-125-daymean-{}.nc"


def load_env_vars():
    """
    Loads environment variables for the runner, raises LookupError if any of
    the environment variables is not loaded.

    :return: dict with environment variables as key value pairs
    """
    var_list = [INPUT_DIR_KEY,
                OUTPUT_DIR_KEY]

    env = dict()
    for var in var_list:
        val = os.getenv(var)
        if var is None:
            raise LookupError("Environment variable {} not found."
                              .format(var))
        env[var] = val

    return env


def process_single_year(input_dir, year):
    """
    Preprocesses a single winter.

    :param input_dir: input directory which contains nc files, must have two
    subdirectories "u" (u component of wind) and "t" (temperature)
    :param year: year which the preprocessed winter starts. Winter timeseries
    are created between October (year) until April (year+1)
    :return: tuple (identifier, p)
    identifier is a string which is the ID of the winter, its format is spe-
    cified by FILE_FORMAT.
    p is a Datapoint object which contains winterly timeseries for each
    variable of interest
    """
    first_winter_u_name = os.path.join(input_dir,
                                       FILE_FORMAT.format("u", "u", year))
    second_winter_u_name = os.path.join(input_dir,
                                        FILE_FORMAT.format("u", "u", year + 1))
    first_winter_t_name = os.path.join(input_dir,
                                       FILE_FORMAT.format("t", "t", year))
    second_winter_t_name = os.path.join(input_dir,
                                        FILE_FORMAT.format("t", "t", year + 1))

    for path in [first_winter_t_name, second_winter_t_name,
                 first_winter_u_name, second_winter_u_name]:
        if not os.path.exists(path):
            raise LookupError("File located at: {} does not exist."
                              .format(path))

    # Create datapoint for the set
    p = RealDataPointFactory.create(first_winter_u_name, second_winter_u_name,
                                    first_winter_t_name, second_winter_t_name)

    # Create identifier for the dataset.
    identifier = "jra55-winter-{}-{}".format(year, year + 1)
    return identifier, p


def process_years(input_dir, output_dir):
    """
    Function to process years
    :param input_dir: Directory which stores nc files
    :param output_dir: Directory where h5 file will be stored
    :return:
    """
    dataset = dict()
    variables = Datapoint.get_variables()
    output_file_path = os.path.join(output_dir, "data_preprocessed.h5")
    out_file = h5py.File(output_file_path, "w")

    # Preprocess each winter
    for year in range(INTERVAL[0], INTERVAL[1]):
        print("Processing year {}-{}".format(year, year + 1))
        identifier, p = process_single_year(input_dir, year)
        dataset[identifier] = p

    print("All years are complete, persisting data...")

    # Persisting data
    for key, data in dataset.items():
        g = out_file.create_group(key)
        for var in variables:
            np_data = getattr(data, var)
            g.create_dataset(var, data=np_data, dtype=np.double)

    out_file.close()
    print("Persisted output in: {}".format(output_file_path))


def run():
    """
    Function to run preprocessing. It loads files from the directory provided
    with the environment variable DSLAB_CLIMATE_BASE_INPUT_REAL and outputs
    preprocesses h5 file to DSLAB_CLIMATE_BASE_OUTPUT_REAL.
    """
    env = load_env_vars()
    input_dir = env[INPUT_DIR_KEY]
    output_dir = env[OUTPUT_DIR_KEY]
    process_years(input_dir, output_dir)


if __name__ == "__main__":
    run()
