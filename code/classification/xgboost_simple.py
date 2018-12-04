import numpy as np
import pandas as pd
import argparse
import functools
from core.data_manager import DataManager
from preprocessing.dataset import DatapointKey as DK
from tsfresh import extract_features
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (accuracy_score, make_scorer, f1_score,
                             roc_auc_score)
from xgboost import XGBClassifier
from matplotlib import pyplot
from utils.set_seed import SetSeed
from utils.output_class import Output
from utils.enums import Task, Classifier, DataType


class ManualAndXGBoost:
    """A class that receives as input the processed data and the definition that
    you want classification for and does classification using the XGBoost
    Classifier.

    Attributes
    ----------
    variables: list
        The initial variables used from the array of variables that we have
    """
    variables = [DK.WIND_60, DK.WIND_65, DK.TEMP_60_90]

    def __init__(self, definition):
        """The constructor of the ManualAndXgboost class. Also sets the random
        seed.

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
        """
        self.definition = definition
        SetSeed().set_seed()

    def __get_labels(self, data_manager):
        """Returns the binary labels as a numpy array for the corresponding
        definition

        Parameters
        ----------
            data_manager: core.DataManager class
                A data manager class initialized with the real or simulated
                data path
        Returns
        -------
            labels: numpy array
                A numpy array of size [num_data x 1] with the labels
        """
        labels = data_manager.get_data_for_variable(self.definition)
        binary_labels = []
        for label in labels:
            binary_labels.append(int(np.any(label[:])))
        labels = np.array(binary_labels)
        return labels

    def _bring_data_to_format(self, data_manager):
        """Gets the corresponding initial variables using the data manager class
        and returns a data numpy array that the variables are in format
        [num_data*num_variables, len_winter]. The trick here is that
        consecutive lines are not of the same variable space but from different
        variable

        Parameters
        ----------
            data_manager: core.DataManager class
                A data manager class initialized with the real or simulated
                data path

        Returns
        -------
            data: numpy array
                A numpy array of format [num_data*num_features, len_winter]
            """
        # returns an array of dimensions (N, FC, D)
        temp_data = data_manager.get_data_for_variables(self.variables)

        data = []
        for i in range(len(temp_data)):
            for j in range(len(temp_data[i])):
                    data.append(temp_data[i, j])

        data = np.array(data)
        return data

    def _produce_features(self, data):
        """
        Gets the data in the format [num_data*num_variables, len_winter] and by
        using tsfresh it produces a matrix [num_data,
        tsfresh_features*num_features]. It also creates a list of
        [tsfresh_features*num_features] to have the correspondence between the
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
                variables_length = len(self.variables)
                for i in range(keys_features_length*variables_length):
                    quotient = int(i / keys_features_length)
                    self.feature_keys.append(
                            self.variables[quotient] + "_" +
                            temp_feature_keys[
                                i-quotient*keys_features_length]
                            )

            features.append(X.values[0])

        # bring the features from format [num_data*num_features, len_winter] to
        # [num_data, num_features*len_winter]
        new_features = []
        length = len(self.variables)
        for i, feature in enumerate(features):
            if i % length == 2:
                new_features.append(np.concatenate((features[i],
                                    features[i-1], features[i-2]),
                                    axis=0))

        features = new_features[:]
        features = np.asarray(features)
        return features

    def preprocess(self, path):
        """This function produces the features and the labels for the simulated
        data. In order to do that it uses other functions to produce the labels
        and the features. It first checks if the labels are in the
        corresponding folder provided as an argument to the class and if not
        it produces them.

        Parameters
        ----------
            path: string
                The input path of the data
        Returns
        -------
            features: numpy array
                A numpy array of shape [num_data x num_features] that contains
                the features calculated by tsfresh
            labels: numpy array
                A numpy array of shape [num_data] that contains the labels
        """
        data_manager = DataManager(path)
        labels = self.__get_labels(data_manager)
        data = self._bring_data_to_format(data_manager)
        features = self._produce_features(data)

        return features, labels

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
        model = XGBClassifier(max_depth=5, n_estimators=1000, reg_alpha=0.1)
        model.fit(X_train, y_train)
        return model

    def evaluate_simulated(self, X, y, scoring):
        """Runs a dictionary of the scores of AUROC, Accuracy and F1-Score
        on 5-CV of the data
        Parameters
        ----------
            X: numpy array
                The numpy array of the data with shape [num_data,
                dimensionality]
            y: numpy array
                The numpy array of the labels with shape [num_data, 1]
            scoring: dict
                A dictionary that has a correspondence from the metrics that we
                use in the output class to the one used by scikit-learn
        Returns
        -------
            scores:
                A python dict with the info provided by the cross_validate
                function of scikit-learn
        """
        model = XGBClassifier(max_depth=5, n_estimators=1000,
                              reg_alpha=0.1)

        # this method will run 5 Stratified CV with a model initialized
        # previously and a scoring dictionary that contains auroc, accuracy and
        # f1 score
        scores = cross_validate(model, X, y, cv=5, scoring=scoring)
        return scores

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

    def test(self, model, X_test, y_test, scoring):
        """Returns the accuracy of the model on the test set.
        Parameters
        ----------
            X_test: numpy array
                The test split of the data features
            y_test: numpy array
                The test split of the data labels

        Returns
        -------
            scores:
                A python dict with the scores in the same format like the
                cross_validate function of scikit-learn
        """
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        score = {}
        for key, value in scoring.items():
            score['test_' + key] = [value(y_test, predictions)]
        return score

    @staticmethod
    def write_to_csv(datatype, scores):
        """Writes to the results.csv file. It accepts the scores as a dictionary
        returned by cross_validate and the datatype (real or simulated) and
        then initializes the Output class with the proper parameters in order
        to write to the .csv file
        Parameters
        ----------
            datatype: utils.enum.DataType instance
                the type of the data (real or simulated)
            scores: dictionary
        """
        for key, value in scores.items():
            if key.startswith('test'):
                results = Output(Classifier.xgboost, Task.classification,
                                 datatype, args.definition, '-', '-', '-',
                                 key.split('_')[1], value)
                results.write_output()


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
            "-sp",
            "--simulated_path",
            help="Choose the input relative path where the simulated data are",
            action="store",
            default="data/simulated_data_labeled.h5"
            )
    parser.add_argument(
            "-rp",
            "--real_path",
            help="Choose the input relative path where the real data are",
            action="store",
            default="data/real_data_labeled.h5"
            )
    parser.add_argument(
            "-dt",
            "--data_type",
            choices=('sim', 'real'),
            help="Choose if the evaluation is going to happen on real or"
                 "simulated data",
            action="store",
            default="sim"
            )
    parser.add_argument(
            "-m",
            "--mode",
            choices=('TT', 'CV'),
            help="Choose the evaluation mode",
            action="store",
            default="TT"
            )
    parser.add_argument(
            "-p",
            "--plot",
            help="Choose if you'll plot the feature importances as bar plots",
            action="store_true",
            default=False
            )
    args = parser.parse_args()
    test = ManualAndXGBoost(
            definition=args.definition
            )

    scoring_sim = {
            'auroc': 'roc_auc',
            'accuracy': make_scorer(accuracy_score),
            'f1': 'f1_macro'
            }
    scoring_real = {
            'auroc': roc_auc_score,
            'accuracy': accuracy_score,
            'f1': functools.partial(f1_score, average='macro')
            }
    if args.data_type == 'sim':
        features, labels = test.preprocess(args.simulated_path)
        if args.mode == 'TT':
            X_train, X_test, y_train, y_test = train_test_split(
                            features, labels, test_size=0.2,
                            stratify=labels)
            model = test.train(X_train, y_train)
            score = test.test(model, X_test, y_test, accuracy_score)
            print("Accuracy: %.2f%%" % (score * 100.0))
            if args.plot:
                test.plot(model)
        else:
            scores = test.evaluate_simulated(features, labels, scoring_sim)
            test.write_to_csv(DataType.simulated, scores)
    else:
        real_features, real_labels = test.preprocess(args.real_path)
        sim_features, sim_labels = test.preprocess(args.simulated_path)
        model = test.train(sim_features, sim_labels)
        scores = test.test(model, real_features, real_labels, scoring_real)
        test.write_to_csv(DataType.real, scores)
