import h5py
import numpy as np
# import sys
import pickle
import pandas as pd
from tsfresh import extract_features


class ManualAndXGBoost:
    matches = {
            'CP07': ['wind_60'],
            'U&T': ['temp_80_90', 'temp_60_70'],
            'U65': ['wind_65'],
            'ZPOL_temp': ['temp_60_90']
            }

    def __init__(self, definition, path, pickling, pickle_path):
        self.definition = definition
        self.path = path
        self.pickling = pickling
        self.pickle_path = pickle_path

    def get_useful_variables(self):
        data = []
        labels = []
        with h5py.File(self.path, 'r') as f:
            for winter, timeseries in f.items():
                for key, value in timeseries.items():
                    if key == self.definition:
                        labels.append(int(np.any(value[:])))
                    if key in self.matches[self.definition]:
                        data.append(np.array(value[:]))
        data = np.array(data)
        labels = np.array(labels)

        return data, labels

    def preprocess(self):
        if self.pickling:
            data, labels = self.get_useful_variables()
            length = data.shape[1]
            features = []
            flag = True
            for i, row in enumerate(data):
                for_tsfresh = pd.DataFrame({'id': np.ones(length),
                                            'time': np.arange(length),
                                            'value': row})
                X = extract_features(for_tsfresh, column_id='id',
                                     column_sort='time')
                if flag:
                    self.feature_keys = X.columns.values
                    flag = False
                features.append(X.values[0])

            self.features = np.array(features)
            with open(str(self.pickle_path), 'wb') as f:
                pickle.dump([features, data, labels], f)
        else:
            with open(str(self.pickle_path), 'rb') as f:
                features, data, labels = pickle.load(f)

    def train(self):
        raise(NotImplementedError)

    def test(self):
        raise(NotImplementedError)


if __name__ == '__main__':
    test = ManualAndXGBoost(
            'CP07', 'data/data_preprocessed_labeled.h5', True,
            'data/data_labels_features.pkl')
    test.preprocess()
