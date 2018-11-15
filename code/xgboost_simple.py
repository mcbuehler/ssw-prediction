import h5py
import numpy as np
import pickle
import pandas as pd
import argparse
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from matplotlib import pyplot


class ManualAndXGBoost:
    """A class that receives as input the processed data and the definition that
    you want classification for and does classification using the XGBoost
    Classifier.

    Attributes
    ----------
    features: list
        The initial features used from the array of features that we have
    """

    features = ['wind_60', 'wind_65', 'temp_60_90']

    def __init__(self, definition, path, pickle_path):
        """the constructor of the ManualAndXgboost class

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
            path: string
                the path where the input data are
            pickle_path: string
                the path where you will store the pickle file that contains the
                labels
        """
        self.definition = definition
        self.path = path
        self.pickle_path = pickle_path

    def get_labels_variables(self, only_labels):
        """Gets the data and according to the definition that the classification
        should be done for extracts the relevant labels and the features

        Parameters
        ----------
            only_labels: boolean
                A flag to decide if you are going to return both labels and
                features or just labels
        Returns
        -------
            data: numpy array
                A numpy array of the relevant features for the definition used
            labels: numpy array
                A numpy array of the relevant labels for the definition used
        """
        data = []
        labels = []

        with h5py.File(self.path, 'r') as f:
            for winter, timeseries in f.items():
                for key, value in timeseries.items():
                    if key == self.definition:
                        labels.append(int(np.any(value[:])))

                    if not only_labels:
                        if key in self.features:
                            data.append(np.array(value[:]))

        data = np.array(data)
        labels = np.array(labels)

        if only_labels:
            return labels
        else:
            return data, labels

    def preprocess(self):
        """This function produces the train and the test split of the features
        as well as the train and the test split of the labels. Multiple things
        are happening. First of all it tries to read the already processed
        features and their labels from tsfresh from a pickle file. If they are
        found available it uses them instead of calculating them again. After
        that is splits the dataset into test and train.

        Returns
        -------
            X_train: numpy array
                A numpy array of shape [num_data x num_features] the training
                data
            X_test: numpy array
                A numpy array of shape [num_data x num_features] the test data
            y_train: numpy array
                A numpy array of shape [num_data x 1] the training labels
            y_test: numpy array
                A numpy array of shape [num_data x 1] the test labels
        """
        try:
            with open(str(self.pickle_path), 'rb') as f:
                features, self.feature_keys = pickle.load(f)
            features = np.array(features)
            labels = self.get_labels_variables(only_labels=True)
        except FileNotFoundError:
            print("Didn't find the .pkl file of the features. Producing it",
                  "now, under the pickle path folder")
            data, labels = self.get_labels_variables(only_labels=False)
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
                    temp_feature_keys = X.columns.values
                    flag = False
                    self.feature_keys = []
                    keys_features_length = len(temp_feature_keys)
                    features_length = len(self.features)
                    for i in range(keys_features_length*features_length):
                        quotient = int(i / keys_features_length)
                        self.feature_keys.append(
                                self.features[quotient] + "_" +
                                temp_feature_keys[
                                    i-quotient*keys_features_length]
                                )

                features.append(X.values[0])

            new_features = []
            for i, feature in enumerate(features):
                if i % 3 == 2:
                    new_features.append(np.concatenate((features[i],
                                        features[i-1], features[i-2]),
                                        axis=0))

            features = new_features[:]
            features = np.asarray(features)
            with open(str(self.pickle_path), 'wb') as f:
                pickle.dump([features, self.feature_keys], f)

        X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2,
                        stratify=labels, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Trains an XGBoostClassifier by getting the training data from other
        parts of the class. Also prints the three most important features for
        the classification and plots them as a bar plot.

        Parameters
        ----------
            X_train: numpy array
                The training split of the data features
            y_train: numpy array
                The training split of the data labels
        Returns
        -------
            model: XGBClassifier class
                The trained model
        """
        model = XGBClassifier()
        model.fit(X_train, y_train)
        top_3 = sorted(model.feature_importances_, reverse=True)[:3]
        max_idxs = []
        for i in range(3):
            for k, j in enumerate(model.feature_importances_):
                if j == top_3[i] and self.feature_keys[k] not in max_idxs:
                    max_idxs.append(self.feature_keys[k])

        print(max_idxs)
        pyplot.title("Feature importance for definition:" + self.definition)
        pyplot.bar(list(range(1, 4)), top_3)
        pyplot.show()
        return model

    def test(self, model, X_test, y_test):
        """Returns the accuracy of the model on the test set.
        Parameters
        ----------
            X_test: numpy array
                The test split of the data features
            y_test: numpy array
                The test split of the data labels

        Returns
        -------
            accuracy: float
                The accuracy of the model
        """
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple classification \
            scheme using feature engineering and the XGBoostClassifier')
    parser.add_argument(
            "-d",
            "--definition",
            choices=('CP07', 'U65', 'ZPOL_temp', 'U&T'),
            help="Choose the definition that you want to run classification",
            action="store",
            default="CP07"
           )
    parser.add_argument(
            "-i",
            "--input_path",
            help="Choose the input path where the data are",
            action="store",
            default="data/data_preprocessed_labeled.h5"
            )
    parser.add_argument(
            "-o",
            "--output_path",
            help="Choose the output path of the pickle file",
            action="store",
            default="data/"
            )
    args = parser.parse_args()
    pickle_path = args.output_path + "features.pkl"
    test = ManualAndXGBoost(
            definition=args.definition,
            path=args.input_path,
            pickle_path=pickle_path
            )
    X_train, X_test, y_train, y_test = test.preprocess()
    model = test.train(X_train, y_train)
    accuracy = test.test(model, X_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
