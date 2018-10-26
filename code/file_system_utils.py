import os

from collections import Iterable


def create_filename(prefix, index, postfix, path_prefix=""):
    """
    Creates a filename in this format: (<path_prefix>/)<prefix><index><postfix>
    :param prefix:
    :param index:
    :param postfix:
    :param path_prefix:
    :return:
    """
    return os.path.join(path_prefix, "{}{}{}".format(prefix, index, postfix))


def get_files_with_prefix(path, prefix, keep_path=False) -> Iterable:
    """
    Returns the folders stored at path that start with prefix.
    :param path:
    :param prefix:
    :param keep_path Prepends path to filename
    :return:
    """
    files = os.listdir(path)
    files_filtered = filter(lambda f: f.startswith(prefix), files)
    if keep_path:
        files_filtered = [os.path.join(path, f) for f in files_filtered]
    return files_filtered


def get_data_file_paths(input_folder, data_file_name) -> Iterable:
    """
    Returns all data files stored in subfolders.
    Subfolders are assumed to be named in the format year_<index>, e.g. "year_82"
    :param input_folder: folder containing subfolders starting with "year_..."
    :param data_file_name: name of file in subfolders, e.g. "atmos_daily.nc"
    :return:
    """
    years = get_files_with_prefix(input_folder, "year_", keep_path=True)
    return [os.path.join(year, data_file_name) for year in years]
