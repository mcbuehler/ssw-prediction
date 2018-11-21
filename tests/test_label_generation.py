import sys

sys.path.append('../code/')

import unittest
import h5py
import numpy as np
import run_label_generation


class TestLabeling(unittest.TestCase):

    def setUp(self):
        f = h5py.File('../data/data_preprocessed.h5', 'r')
        data_fields = \
            ['temp_60_90', 'temp_60_70', 'temp_80_90', 'wind_60', 'wind_65']
        keys = list(f.keys())

        self.data = np.array(
            [[f[key][data_field] for data_field in data_fields] for key in keys[:10]])

    def test_UnT(self):
        masks = run_label_generation.UnT(self.data)

        for i, mask in enumerate(masks):
            # The first day of SSW has reversed wind
            self.assertTrue(all(self.data[i, 3, :][mask] < 0))
            # One day before, wind is not reversed
            mask2 = np.roll(mask, -1)
            self.assertTrue(all(self.data[i, 3, :][mask2] > 0))

    def test_CP07(self):
        masks = run_label_generation.CP07(self.data)

        for i, mask in enumerate(masks):
            # The first day of SSW has reversed wind
            self.assertTrue(all(self.data[i, 3, :][mask] < 0))
            # One day before, wind is not reversed
            mask2 = np.roll(mask, -1)
            self.assertTrue(all(self.data[i, 3, :][mask2] > 0))

    def test_U65(self):
        masks = run_label_generation.U65(self.data)

        for i, mask in enumerate(masks):
            # The first day of SSW has reversed wind
            self.assertTrue(all(self.data[i, 4, :][mask] < 0))
            # One day before, wind is not reversed
            mask2 = np.roll(mask, -1)
            self.assertTrue(all(self.data[i, 3, :][mask2] > 0))


if __name__ == '__main__':
    unittest.main()
