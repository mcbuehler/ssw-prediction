import argparse
import functools
import numpy as np
from xgboost import XGBClassifier
from classification.xgboost_simple import ManualAndXGBoost
from prediction_set import FixedWindowPredictionSet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from utils.set_seed import SetSeed
from imblearn.over_sampling import ADASYN
from utils.enums import DataType, Task, Classifier
from utils.output_class import Output


class XGBoostPredict(ManualAndXGBoost):
    """A class that receives as input the processed data and the definition that
    you want prediction for and does prediction using the XGBoost Classifier.
    It inherits from the ManualAndXGBoost class that does classification in
    order to avoid code repetition. It also uses the PredictionSet class to
    create the labels and the train/test set.
    """

    def __init__(self, definition, cutoff_point, features_interval,
                 prediction_interval):
        """The constructor of the XGBoostPredict class
        Parameters
        ----------
            definition: string
                The definition that you will get the labels for
            cutoff_point: int
                The maximum cutoff_point where you will look your time series
            features_interval: int
                The number of days in the past that you will look the time
                series before the cutoff_point
            prediction_interval: int
                The interval that you will predict in the future
        """
        self.definition = definition
        self.cutoff_point = cutoff_point
        self.features_interval = features_interval
        self.prediction_interval = prediction_interval

        # set the seed for all the libraries
        SetSeed().set_seed()

    def _resample(self, data, labels):
        """Resamples the data using the ADASYN algorithm.

        Parameters
        ----------
            data: numpy array
                An array of the form [num_data, variable_count*dimensionality]
            labels: numpy array
                An array of the form [num_data, 1]

        Returns
        -------
            X_train: numpy array
                An array of the form [num_resampled_data*variable_count,
                dimensionality]
            y_train: numpy array
                An array of the form [num_resampled_data, 1]
        """
        X_train, y_train = ADASYN().fit_resample(data, labels)
        return X_train, y_train

    def _stack_variables(self, temp_data):
        """Brings data to the right format for resampling. More specifically
        it gets a 3D array of dimensions of (N, FC, D) and returns a 2D array
        of dimensions (N*FC, D) where 3 consecutive lines belong in different
        features

        Parameters
        -------
            temp_data: np.array
                a numpy array of dimensions (N, FC, D)
        Returns
        -------
            data: np.array
                a numpy array of dimensions (N*FC, D)
        """
        self.feature_count = temp_data.shape[1]
        data = []
        for i in range(len(temp_data)):
            for j in range(len(temp_data[i])):
                if j % self.feature_count == self.feature_count - 1:
                    data.append(np.hstack((temp_data[i, j],
                                           temp_data[i, j-1],
                                           temp_data[i, j-2])))

        return np.array(data)

    def _split_variables(self, temp_data):
        """Takes the data in the form where each row has all the variables
        stacked horizontally and split each row into num_variables rows

        Parameters
        ----------
            temp_data: numpy array
                An array of the form [num_data, variable_count*dimensionality]

        Returns
        -------
            data: numpy array
                An array of the form [num_data*variable_count, dimensionality]
        """
        data = [array for row in temp_data for array in
                np.split(row, self.feature_count)]

        return np.array(data)

    def get_data_and_labels(self, path):
        """This function uses the FixedWindowPredictionSet class in order to
        return the data and the labels given the parameters.

        Parameters
        ----------
            path: string
                The path of the data (real or simulated)
            definition: string
                The definition that you will get the labels for
            cutoff_point: int
                The maximum cutoff_point where you will look your time series
            features_interval: int
                The number of days in the past that you will look the time
                series before the cutoff_point
            prediction_interval: int
                The interval that you will predict in the future
        Returns
        -------
            data: numpy array
                A numpy array of shape [num_data, num_features, dimensionality]
                that contains the data
            labels: numpy array
                A numpy array of shape [num_data] that contains the labels
        """
        self.prediction_set = FixedWindowPredictionSet(
                self.definition,
                path,
                self.cutoff_point,
                self.prediction_interval,
                self.features_interval)

        labels = np.ravel(self.prediction_set.get_labels())

        # returns data in format (N, FC, D)
        data = self.prediction_set.get_features()
        return data, labels

    def tune_classifier(self, X_train, y_train):
        """Finetunes the XGBoostClassifier for better ROCAUC
        Parameters
        ----------
            X_train: numpy array
                A numpy array of shape [num_data x num_features] the training
                data
            X_test: numpy array
                A numpy array of shape [num_data x num_features] the test data
            y_train: numpy array
                A numpy array of shape [num_data x 1] the training labels

        """
        tuned_parameters = {
                'max_depth': [3, 5, 10],
                'n_estimators': [500, 1000, 2000],
                'reg_alpha': [0, 0.1, 5, 10, 100],
                'reg_lambda': [0, 0.1, 5, 10, 100]
                }
        clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=5,
                           scoring='roc_auc')
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

    def train(self, X_train, y_train):
        """Trains an XGBoostClassifier by getting the training data from other
        parts of the class.

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

        model = XGBClassifier(n_estimators=1000, max_depth=5, reg_alpha=0.1)
        model.fit(X_train, y_train)
        return model

    def pipeline(self, X_train, y_train, X_test, y_test):
        """A method to pipeline the steps that need to be done in train and
        test. First oversamples the training set and then calculates features
        from the oversampled training set and the test set. After that it
        trains a classifier on the training set and tests on the test set
        Parameters
        ----------
            X_train: numpy array
                The train split of the data features
            y_train: numpy array
                The train split of the data labels
            X_test: numpy array
                The test split of the data features
            y_test: numpy array
                The test split of the data labels

        Returns
        -------
            scores: dict
                A python dict with the scores in the same format like the
                cross_validate function of scikit-learn
        """
        X_train, y_train = self._resample(X_train, y_train)
        X_train = self._split_variables(X_train)
        X_test = self._split_variables(X_test)
        self.feature_keys, X_train = super()._produce_features(
                X_train, self.variables)
        _, X_test = super()._produce_features(X_test, self.variables)
        model = self.train(X_train, y_train)
        temp_scores = super().test(model, X_test, y_test, scoring)
        return temp_scores

    def evaluate_simulated(self, X, y, scoring):
        """Returns a dictionary of the scores of AUROC, Accuracy and F1-Score
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

        # this method will run 5 fold StratifiedCV with a scoring dictionary
        # that contains auroc, accuracy and f1 score
        skf = StratifiedKFold(n_splits=5)
        X = self._stack_variables(X)
        scores = {}
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            temp_scores = self.pipeline(X_train, y_train, X_test, y_test)
            for key, value in temp_scores.items():
                try:
                    scores[key].append(value[0])
                except KeyError:
                    scores[key] = value
        return scores

    def write_to_csv(self, datatype, scores):
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
                results = Output(Classifier.xgboost, Task.prediction,
                                 datatype, self.definition, self.cutoff_point,
                                 self.features_interval,
                                 self.prediction_interval,
                                 key.split('_')[1], value)
                results.write_output()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A prediction scheme \
            using feature engineering and the XGBoostClassifier')
    parser.add_argument(
            "-d",
            "--definition",
            choices=('CP07', 'U65', 'U&T'),
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
            default="CV"
            )
    parser.add_argument(
            "-cp",
            "--cutoff_point",
            help="Choose the cutoff point of the time series",
            type=int,
            action="store",
            default=90
            )
    parser.add_argument(
            "-fi",
            "--features_interval",
            help="Choose the interval where you will calculate features",
            type=int,
            action="store",
            default=30
            )
    parser.add_argument(
            "-pi",
            "--prediction_interval",
            help="Choose the max prediction interval",
            type=int,
            action="store",
            default=5
            )
    args = parser.parse_args()
    scoring = {
            'auroc': roc_auc_score,
            'accuracy': accuracy_score,
            'f1': functools.partial(f1_score, average='macro')
            }
    test = XGBoostPredict(
            definition=args.definition,
            cutoff_point=args.cutoff_point,
            features_interval=args.features_interval,
            prediction_interval=args.prediction_interval
            )
    if args.data_type == 'sim':
        data, labels = test.get_data_and_labels(args.simulated_path)
        if args.mode == 'TT':
            data = test._stack_variables(data)
            X_train, X_test, y_train, y_test = train_test_split(
                            data, labels, test_size=0.2,
                            stratify=labels)

            scores = test.pipeline(X_train, y_train, X_test, y_test)
            test.write_to_csv(DataType.simulated, scores)
        else:
            scores = test.evaluate_simulated(data, labels, scoring)
            test.write_to_csv(DataType.simulated, scores)
    else:
        real_data, real_labels = test.get_data_and_labels(args.real_path)
        sim_data, sim_labels = test.get_data_and_labels(args.simulated_path)
        real_data = test._stack_variables(real_data)
        sim_data = test._stack_variables(sim_data)
        scores = test.pipeline(sim_data, sim_labels, real_data, real_labels)
        test.write_to_csv(DataType.real, scores)
