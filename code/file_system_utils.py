import os

from collections import Iterable
from typing import Union

import re


def create_filename(prefix: str, index: Union[int, str], postfix: str,
                    path_prefix: str = ""):
    """
    Creates a filename in this format: (<path_prefix>/)<prefix><index><postfix>
    :param prefix:
    :param index:
    :param postfix:
    :param path_prefix:
    :return:
    """
    return os.path.join(path_prefix, "{}{}{}".format(prefix, index, postfix))


def reorder_numerically(file_list: Iterable) -> list:
    """
    Returns an ordered list of the strings in file_list (ascending).
    The list is ordered by the first positive numerical value found in the
    entries of file_list. Entries without numerical value are returned first.
    :param file_list:
    :return: list
    """

    def sort_function(filename):
        numbers = re.findall(r'\d+', filename)
        print(numbers)
        if len(numbers):
            return int(numbers[0])
        return 0

    file_list_sorted = sorted(file_list, key=sort_function)
    return file_list_sorted


def get_files(path: str, prefix: str = "", postfix: str = "",
              keep_path: bool = False,
              order_numerical: bool = True) -> Iterable:
    """
    Returns the fiels stored at path that start with
    prefix and end with postfix.
    :param path:
    :param prefix:
    :param postfix
    :param keep_path: Prepends path to filename
    :param order_numerical: Returns files sorted by the first integer
    value they contain.
    :return:
    """
    files = os.listdir(path)
    files_filtered = filter(
        lambda f: f.startswith(prefix) and f.endswith(postfix), files)
    if order_numerical:
        files_filtered = reorder_numerically(files_filtered)
    print("Processing files in the following order:", files_filtered)
    if keep_path:
        files_filtered = [os.path.join(path, f) for f in files_filtered]
    return files_filtered


def get_data_file_paths(input_folder: str, data_file_prefix: str = "",
                        data_file_postfix: str = ".nc") -> list:
    """
    Returns all data files stored in subfolders.
    Subfolders are assumed to be named in the format year_<index>,
     e.g. "year_82"
    :param input_folder: folder containing subfolders starting
    with "year_..." or "daymean"
    :param data_file_prefix: prefix of data file, e.g. "atmos_daymean_"
    :param data_file_postfix: postfix of data file, e.g. ".nc"
    :return:
    """
    data_file_paths = list()
    # we look at all subfolders, whatever they are named
    subfolders = get_files(input_folder, keep_path=True, order_numerical=True)
    for subfolder in subfolders:
        data_files = get_files(subfolder, prefix=data_file_prefix,
                               postfix=data_file_postfix, keep_path=True,
                               order_numerical=True)
        data_file_paths += data_files
    return data_file_paths
