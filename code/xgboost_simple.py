import h5py
import numpy as np
# import sys
import pickle
import pandas as pd
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class ManualAndXGBoost:
    matches = {
            'CP07': ['wind_60'],
            'U&T': ['temp_80_90', 'temp_60_70', 'wind_60'],
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

        if self.definition == 'U&T':
            new_data = []
            for i in range(len(data)):
                if i % 3 == 2:
                    new_data.append(data[i])
                elif i % 3 == 1:
                    new_data.append(data[i] - data[i-1])
            data = new_data[:]

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

            if self.definition == 'U&T':
                new_features = []
                for i, feature in enumerate(features):
                    if i % 2 == 1:
                        new_features.append(np.concatenate((features[i],
                                            features[i-1]), axis=0))

                features = new_features[:]
            features = np.asarray(features)
            with open(str(self.pickle_path), 'wb') as f:
                pickle.dump([features, data, labels], f)
        else:
            with open(str(self.pickle_path), 'rb') as f:
                features, data, labels = pickle.load(f)
            features = np.array(features)

        print(np.unique(labels, return_counts=True))
        print(type(features), type(labels))
        X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2,
                        stratify=labels, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        model = XGBClassifier()
        model.fit(X_train, y_train)
        max_imp = max(model.feature_importances_)
        max_idxs = [i for i, j in enumerate(model.feature_importances_)
                    if j == max_imp]
        print(self.feature_keys[max_idxs[0]])
        return model

    def test(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        return accuracy


if __name__ == '__main__':
    test = ManualAndXGBoost(
            'U&T', 'data/data_preprocessed_labeled.h5', True,
            'data/data_labels_features_U&T.pkl')
    X_train, X_test, y_train, y_test = test.preprocess()
    model = test.train(X_train, y_train)
    accuracy = test.test(model, X_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
