import os
import unittest

import numpy as np
import run_label_generation
from data_manager import DataManager
from dataset import DatapointKey


class TestLabeling(unittest.TestCase):

    def setUp(self):

        path_preprocessed = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT")
        path_in = os.path.join(path_preprocessed, "data_preprocessed.h5")

        # Create a datamanager for input data.
        data_manager = DataManager(path_in)

        # Get group names and dictionary names
        data_fields = [DatapointKey.TEMP_60_90,
                       DatapointKey.TEMP_60_70,
                       DatapointKey.TEMP_80_90,
                       DatapointKey.WIND_60,
                       DatapointKey.WIND_65]

        self.data = {data_field: data_manager.get_data_for_variable(data_field)
                     for data_field in data_fields}

    def test_UnT(self):

        data = self.data
        number_of_years = len(data[DatapointKey.WIND_60])
        winters = [{data_field: data[data_field][i, :] for data_field
                    in data.keys()} for i in range(number_of_years)]

        masks = [run_label_generation.UnT(xi) for xi in winters]

        for i, mask in enumerate(masks):
            # The first day of SSW has reversed wind
            self.assertTrue(all(data[DatapointKey.WIND_60][i, :][mask] < 0))
            # One day before, wind is not reversed
            mask2 = np.roll(mask, -1)
            self.assertTrue(all(data[DatapointKey.WIND_60][i, :][mask2] > 0))

    def test_CP07(self):

        data = self.data
        number_of_years = len(data[DatapointKey.WIND_60])
        winters = [{data_field: data[data_field][i, :] for data_field
                    in data.keys()} for i in range(number_of_years)]

        masks = [run_label_generation.CP07(xi) for xi in winters]

        for i, mask in enumerate(masks):
            # The first day of SSW has reversed wind
            self.assertTrue(all(data[DatapointKey.WIND_60][i, :][mask] < 0))
            # One day before, wind is not reversed
            mask2 = np.roll(mask, -1)
            self.assertTrue(all(data[DatapointKey.WIND_60][i, :][mask2] > 0))

    def test_U65(self):

        data = self.data
        number_of_years = len(data[DatapointKey.WIND_60])
        winters = [{data_field: data[data_field][i, :] for data_field
                    in data.keys()} for i in range(number_of_years)]

        masks = [run_label_generation.U65(xi) for xi in winters]

        for i, mask in enumerate(masks):
            # The first day of SSW has reversed wind
            self.assertTrue(all(data[DatapointKey.WIND_65][i, :][mask] < 0))
            # One day before, wind is not reversed
            mask2 = np.roll(mask, -1)
            self.assertTrue(all(data[DatapointKey.WIND_65][i, :][mask2] > 0))


if __name__ == '__main__':
    unittest.main()
