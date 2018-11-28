import argparse
# import sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from classification.xgboost_simple import ManualAndXGBoost
from prediction_set import FixedWindowPredictionSet
from sklearn.model_selection import train_test_split, GridSearchCV
from utils.set_seed import SetSeed
from imblearn.over_sampling import ADASYN


class XGBoostPredict(ManualAndXGBoost):
    """A class that receives as input the processed data and the definition that
    you want prediction for and does prediction using the XGBoost Classifier.
    It inherits from the ManualAndXGBoost class that does classification in
    order to avoid code repetition. It also uses the PredictionSet class to
    create the labels and the train/test set.
    """

    def __init__(self, definition, path, cutoff_point,
                 features_interval, prediction_interval):
        """The constructor of the XGBoostPredict class

        Parameters
        ----------
            definition: string
                the definition that you want to do classification for
            path: string
                the path where the input data are
            cutoff_point: int
                the maximum cutoff point until where your data will expand
            prediction_interval: int
                the interval where you will do predictions in
            features_interval: int
                the interval in the past where you will produce features
        """

        self.prediction_interval = prediction_interval
        self.prediction_set = FixedWindowPredictionSet(definition, path,
                                                       cutoff_point,
                                                       prediction_interval,
                                                       features_interval)
        # set the seed for all the libraries
        SetSeed().set_seed()

    def _stack_variables_and_resample(self, temp_data, labels):
        """Brings data to the right format for features engineering and training
        a classifier. More specifically it gets a 3D array of dimensions of (N,
        FC, D) and returns a 2D array of dimensions (N*FC, D) where 3
        consecutive lines belong in different features


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

        X_train, X_test, y_train, y_test = train_test_split(
                        data, labels, test_size=0.2,
                        stratify=labels)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)

        return X_train, y_train, X_test, y_test

    def _split_variables(self, temp_data):
        data = []
        for row in temp_data:
            split_variables = np.split(row, self.feature_count)
            for array in split_variables:
                data.append(array)

        data = np.array(data)
        return data

    def preprocess_as_prediction(self):
        """This function produces the train and the test split of the features
        as well as the train and the test split of the labels. In order to do
        that it uses other functions to produce the labels and the features. It
        first checks if the labels are in the corresponding folder provided as
        an argument to the class and if not it produces them. Then it does the
        splitting.

        Returns
        -------
            X_train: numpy array
                A numpy array of shape [num_data x num_features] that contains
                the training data
            X_test: numpy array
                A numpy array of shape [num_data x num_features] that contains
                the test data
            y_train: numpy array
                A numpy array of shape [num_data] that contains the training
                labels
            y_test: numpy array
                A numpy array of shape [num_data] that contains the test labels
        """
        labels = np.ravel(self.prediction_set.get_labels())
        # get distribution of the labels
        # print(np.unique(labels[:], return_counts=True))
        # sys.exit(0)

        # returns data in format (N, FC, D)
        data = self.prediction_set.get_features()
        X_train, y_train, X_test, y_test = \
            self._stack_variables_and_resample(data, labels)
        X_train = self._split_variables(X_train)
        X_test = self._split_variables(X_test)
        X_train = self._produce_features(X_train)
        X_test = self._produce_features(X_test)

        return X_train, X_test, y_train, y_test

    def tune_classifier(self, X_train, y_train):
        """Finetunes the XGBoostClassifier for better accuracy
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

        model = XGBClassifier(n_estimators=1000, max_depth=5, reg_alpha=0.1)
        model.fit(X_train, y_train)
        return model

    def test(self, model, X_test, y_test):
        """Returns the AUROC and F1 score of the model on the test set.
        Parameters
        ----------
            X_test: numpy array
                The test split of the data features
            y_test: numpy array
                The test split of the data labels

        Returns
        -------
            auroc: float
                The AUROC score of the model
            f1: float
                The F1 score of the model
        """

        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        return auc, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A prediction scheme \
            using feature engineering and the XGBoostClassifier')
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
    test = XGBoostPredict(
            definition=args.definition,
            path=args.input_path,
            cutoff_point=args.cutoff_point,
            features_interval=args.features_interval,
            prediction_interval=args.prediction_interval
            )
    X_train, X_test, y_train, y_test = test.preprocess_as_prediction()
    model = test.train(X_train, y_train)
    auc, f1 = test.test(model, X_test, y_test)
    output_string = "Max cutoff: {0} days, Look in the past: {1} days "\
        "Prediction {2} days, AUROC: {3:.2f}, F1:{4:.2f}"
    print(output_string.format(args.cutoff_point, args.features_interval,
          args.prediction_interval, auc, f1))
