import numpy as np
import pickle
import pandas as pd
import argparse
from core.data_manager import DataManager
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
        """The constructor of the ManualAndXgboost class

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
        self.data_manager = DataManager(self.path)

    def __get_labels(self):
        """Returns the binary labels as a numpy array for the corresponding
        definition

        Returns
        -------
            labels: numpy array
                A numpy array of size [num_data x 1] with the labels
        """
        labels = self.data_manager.get_data_for_variable(self.definition)
        binary_labels = []
        for label in labels:
            binary_labels.append(int(np.any(label[:])))
        labels = np.array(binary_labels)
        return labels

    def _bring_data_to_format(self):
        """Gets the corresponding initial variables using the data manager class
        and returns a data numpy array that the features in format
        [num_data*num_features, len_winter]. The trick here is that
        consecutive lines are not of the same feature space but from different
        features
        Returns
        -------
            data: numpy array
                A numpy array of format [num_data*num_features, len_winter]
            """
        temp_data = []
        for feature in self.features:
            temp_data.append(self.data_manager.get_data_for_variable(
                             feature))

        temp_data = np.array(temp_data)
        num_variables = len(self.features)
        num_data = temp_data[0].shape[0]
        data = []
        # The tricky part here is that now we have a 3D tensor for the three
        # variables but it is actually better to convert it to a 2D tensor that
        # will have the 3 features in 3 consecutive lines instead of having
        # them with distance num_data (in our case ~3000)
        for i in range(num_data):
            for j in range(num_variables):
                data.append(temp_data[j, i, :])
        data = np.array(data)
        # testing that the transformation was done correctly
        # for i in range(num_data*3):
        #     print(np.array_equal(data[i], temp_data[i % 3][int(i/3)]))
        # sys.exit(0)
        return data

    def _produce_features(self, data):
        """
        Gets the data in the format [num_data*num_features, len_winter] and by
        using tsfresh it produces a matrix [num_data,
        tsfresh_features*num_variables]. It also creates a list of
        [tsfresh_features*num_variables] to have the correspondence between the
        initial variables and the features produced and saves that in a class
        variable.

        Returns
        -------
            features: np.array
                A numpy array of size
                [num_data, tsfresh_features*num_variables]
        """
        length = data.shape[1]
        features = []
        flag = True
        # iterate though all the data points
        for i, row in enumerate(data):
            # bring them into the format the tsfresh wants them to compute the
            # features
            for_tsfresh = pd.DataFrame({'id': np.ones(length),
                                        'time': np.arange(length),
                                        'value': row})
            X = extract_features(for_tsfresh, column_id='id',
                                 column_sort='time')
            # in the first iteration store the corresponding names of the the
            # features as well. You have to expand the features for one
            # variable to prepend the name of the variable and also do that
            # for all the variables.
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

        # bring the features from format [num_data*num_features, len_winter] to
        # [num_data, num_features*len_winter]
        new_features = []
        for i, feature in enumerate(features):
            if i % 3 == 2:
                new_features.append(np.concatenate((features[i],
                                    features[i-1], features[i-2]),
                                    axis=0))

        features = new_features[:]
        features = np.asarray(features)
        return features

    def preprocess(self):
        """This function produces the train and the test split of the features
        as well as the train and the test split of the labels. In order to do
        that it uses other functions to produce the labels and the features. It
        first checks if the labels are in the corresponding folder provided as
        an argument to the class and if not it produces them. Then it does the
        splitting.

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
        labels = self.__get_labels()
        try:
            with open(str(self.pickle_path), 'rb') as f:
                features, self.feature_keys = pickle.load(f)
            features = np.array(features)
        except FileNotFoundError:
            print("Didn't find the .pkl file of the features. Producing it",
                  "now, under the pickle path folder")

            data = self._bring_data_to_format()
            features = self._produce_features(data)
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
        return model

    def plot(self, model):
        """ Prints the three most important features for the classification
        when the XGB Classifier has been used and plots them as a bar plot.
        Parameters
        ----------
            model: XGBClassifier class
                A trained model
        """

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
            help="Choose the input relative path where the data are",
            action="store",
            default="data/data_labeled.h5"
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
    test.plot(model)
