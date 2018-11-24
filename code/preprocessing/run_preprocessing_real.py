import os
from preprocessor_real import RealDataPointFactory
from dataset import Datapoint
import h5py
import numpy as np

input_dir_key = "DSLAB_CLIMATE_BASE_INPUT_REAL"
output_dir_key = "DSLAB_CLIMATE_BASE_OUTPUT_REAL"
interval_of_interest = (1958, 2016)
file_format = "/{}/{}-jra55-125-daymean-{}.nc"


def load_env_vars():
    """
    Loads environment variables for the runner, raises LookupError if any of
    the environment variables is not loaded.

    :return: dict with environment variables as key value pairs
    """
    var_list = [input_dir_key,
                output_dir_key]

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

    :param input_dir:
    :param year:
    :return:
    """
    first_winter_u_name = input_dir + file_format.format("u", "u", year)
    second_winter_u_name = input_dir + file_format.format("u", "u", year + 1)
    first_winter_t_name = input_dir + file_format.format("t", "t", year)
    second_winter_t_name = input_dir + file_format.format("t", "t", year + 1)

    for path in [first_winter_t_name, second_winter_t_name,
                 first_winter_u_name, second_winter_u_name]:
        if not os.path.exists(path):
            raise LookupError("File located at: {} does not exist."
                              .format(path))

    p = RealDataPointFactory.create(first_winter_u_name, second_winter_u_name,
                                    first_winter_t_name, second_winter_t_name)

    # Create identifier for the dataset.
    identifier = "jra55-winter-{}-{}".format(year, year + 1)
    return identifier, p


def process_years(input_dir, output_dir):
    dataset = dict()
    variables = Datapoint.get_variables()
    output_file_path = os.path.join(output_dir, "data_preprocessed.h5")
    out_file = h5py.File(output_file_path, "w")

    for year in range(interval_of_interest[0], interval_of_interest[1]):
        print("Processing year {}-{}".format(year, year + 1))
        identifier, p = process_single_year(input_dir, year)
        dataset[identifier] = p

    print("All years are complete, persisting data...")

    for key, data in dataset.items():
        g = out_file.create_group(key)
        for var in variables:
            np_data = getattr(data, var)
            g.create_dataset(var, data=np_data, dtype=np.double)

    out_file.close()
    print("Persisted output in: {}".format(output_file_path))


def run():
    env = load_env_vars()
    input_dir = env[input_dir_key]
    output_dir = env[output_dir_key]
    process_years(input_dir, output_dir)


if __name__ == "__main__":
    run()
