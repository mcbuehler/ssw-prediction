import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, \
    accuracy_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from skorch import NeuralNetClassifier

import cnn_model
from data_manager import DataManager
from dataset import DatapointKey as DPK
from set_gpu import set_gpu
from set_seed import SetSeed
from utils.enums import Classifier, Task, Metric, DataType
from utils.output_class import Output


class CNNClassification():
    """
    Class which encapsulates Pytorch Neural Networks model to train, load and
    save models contained at cnn_model.py
    """

    def __init__(self, path_train, definition, path_test=None,
                 c_model_name=Classifier.cnn, cv_folds=5, num_epochs=100,
                 batch_size=8, learning_rate=0.0004):
        """
        Initializer for CNNClassification class.

        :param path_train: path to the h5 file for the training dataset
        (simulated data)
        :param definition: Type of SSW definition to use. example: "CP07",
        "U65"
        :param path_test: path to the h5 file for the test dataset (real data)
        :param c_model_name: CNN classifier name which is going to be used
        example: "cnn", "cnn_max_pool"
        :param cv_folds: number of folds for the cross-validation which is
        used to evaluate the performance on the training dataset.
        :param num_epochs: Number of epochs to train the model
        :param batch_size: Batch size for Adam optimizer
        :param learning_rate: Learning rate for Adam optimizer
        """

        # Device configuration
        device = torch.device('cuda:' + os.getenv("CUDA_VISIBLE_DEVICES")
                              if torch.cuda.is_available() else 'cpu')

        SetSeed().set_seed()

        self.data_manager_train = DataManager(path_train)
        if path_test:
            self.data_manager_test = DataManager(path_test)

        self.definition = definition

        self.cv_folds = cv_folds

        self.metric_txt = ["F1", "ROCAUC", "Accuracy"]
        self.metrics = [f1_score, roc_auc_score, accuracy_score]
        self.model_class_name = c_model_name

        # Number of channels in the CNN - number of features to use
        num_ts = 3

        self.classifier = NeuralNetClassifier(
            cnn_model.get_cnn_classes()[c_model_name](num_ts),
            criterion=nn.CrossEntropyLoss,
            max_epochs=num_epochs,
            lr=learning_rate,
            batch_size=batch_size,
            device=device,
            optimizer=torch.optim.Adam,
        )

    def get_data(self, test=False,
                 variables=(DPK.WIND_65, DPK.TEMP_60_90, DPK.WIND_60)):
        """
        Loads training or test data using DataManager class.

        :param test: bool which indicates whether to get test dataset or
        training dataset
        :param variables: list of identifiers for the variables to be loaded
        :return: (X,y): X is the design matrix for the training or test data
        with type (np.float32). y is the list of labels for the data with type
        (np.int6)
        """

        if test:
            data_manager = self.data_manager_test
        else:
            data_manager = self.data_manager_train

        X = data_manager.get_data_for_variables(variables)
        y = np.any(data_manager.get_data_for_variable(definition),
                   axis=1)
        y = y.astype(np.int64)
        X = X.astype(np.float32)
        return X, y

    def cv_evaluate_simulated(self):
        """
        Evaluates performance on the simulated (training) data by using
        cross-validation.

        :return: a tuple which consists of two lists:
        a list of metrics which indicate performance of the
        classifier which is trained on simulated data and evaluated on the
        real data.
        a list standard deviations of these metrics, estimated by
        cross-validation.
        """

        # We have an unbalanced dataset, so we stratify
        cv = StratifiedKFold(self.cv_folds, shuffle=True)
        X, y = self.get_data()

        scorers = {txt: make_scorer(metric) for txt, metric
                   in zip(self.metric_txt, self.metrics)}

        # Produce scores for all scoring metrics
        scores = cross_validate(self.classifier, X, y, cv=cv,
                                scoring=scorers)

        # We only want to keep mean and std for each metric
        scores_means = [np.mean(scores['test_' + score_type])
                        for score_type in self.metric_txt]

        scores_std = [np.std(scores['test_' + score_type])
                      for score_type in self.metric_txt]

        return scores_means, scores_std

    def evaluate_real(self, train=True):
        """
        Evaluates the performance on the real (test) data.

        :param train: bool which indicates whether to train the classifier
        before evaluation
        :return: a list of metrics which indicate performance of the
        classifier which is trained on simulated data and evaluated on the
        real data
        """

        if self.data_manager_test is None:
            raise Exception("Data manager for the test dataset is not "
                            "available, please initialize")

        # Get train data
        X, y = self.get_data()

        if train is True:
            self.classifier.fit(X, y)

        # Get test data
        X_test, y_test = self.get_data(test=True)

        # Predict real test dataset and evaluate
        y_pred = self.classifier.predict(X_test)

        scores = [metric(y_test, y_pred) for metric in self.metrics]

        return scores

    def write_output(self, scores, data_type):
        """
        Writes results by using the utils.output_class.Output class

        :param scores: a list of scores with order: [f1, auroc, accuracy]
        :param data_type: a string to indicate the type of the data which is
        evaluated. example: "real", "simulated"
        """

        # Write results to output file
        output_default_args = dict(
            classifier=self.model_class_name,
            task=Task.classification,
            data_type=data_type,
            definition=self.definition,
            cutoff_point="-",
            feature_interval="-",
            prediction_interval="-"
        )

        out_metrics = (
            (Metric.f1, scores[0]),
            (Metric.auroc, scores[1]),
            (Metric.accuracy, scores[2])
        )

        for metric, score in out_metrics:
            Output(metric=metric, scores=[score],
                   **output_default_args).write_output()

    def save_model(self):
        """
        Saves the CNN model (skorch.NeuralNetClassifier object). Uses
        environment variable "CNN_WEIGHTS" as the directory and saves the
        file with format
        """
        time_str = time.strftime("%Y%m%dT%H%M%S")
        dir_output = os.getenv("CNN_WEIGHTS")

        if dir_output is None:
            print("Your output path is not set correctly. Please set the "
                  "CNN_WEIGHTS env variable.")
        else:
            file_name = "{}-{}-{}.pkl".format(self.model_class_name,
                                              self.definition, time_str)
            path_output = os.path.join(dir_output, file_name)
            self.classifier.save_params(f_params=path_output)

    def load_model(self, path_in):
        """
        Loads the CNN model (skorch.NeuralNetClassifier object).

        :param path_in: path to load the model
        """
        self.classifier.initialize()
        self.classifier.load_params(f_params=path_in)


def run_classification(path_train, definition, path_test, c_model_name,
                       save=False):
    """

    :param path_train: path to the h5 file for the training dataset
    (simulated data)
    :param definition: Type of SSW definition to use. example: "CP07",
    "U65"
    :param path_test: path to the h5 file for the test dataset (real data)
    :param c_model_name: CNN classifier name which is going to be used
    example: "cnn", "cnn_max_pool"
    :param save: bool which indicated whether to save the model after
    evaluation or not
    :return:
    """
    cl = CNNClassification(path_train, definition, path_test=path_test,
                           c_model_name=c_model_name)

    scores_sim, std_sim = cl.cv_evaluate_simulated()
    scores_real = cl.evaluate_real()

    print(scores_real)
    print(scores_sim)

    cl.write_output(scores_real, DataType.real)
    cl.write_output(scores_sim, DataType.simulated)

    if save:
        cl.save_model()

    return dict(scores_sim=scores_sim,
                scores_real=scores_real,
                std_sim=std_sim)


if __name__ == '__main__':
    set_gpu()

    # Path for training data
    path_preprocessed = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT")
    path_train = os.path.join(path_preprocessed,
                              "data_labeled.h5")

    # Path for testing data
    path_preprocessed_real = os.getenv("DSLAB_CLIMATE_BASE_OUTPUT_REAL")
    path_test = os.path.join(path_preprocessed_real,
                             "data_labeled.h5")

    # Definitions to look
    definitions = [DPK.CP07, DPK.UT, DPK.U65]

    # For each definition run a classifier, get 5-fold CV result for simulate
    # data. Also get the test results.
    for definition in definitions:
        for c_model_name, c_model in cnn_model.get_cnn_classes().items():
            run_classification(path_train, definition, path_test,
                               c_model_name)
