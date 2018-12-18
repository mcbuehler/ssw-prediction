import argparse
import functools
import torch
from torch.optim import SGD
from prediction.xgboost_prediction import XGBoostPredict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from utils.enums import DataType, Classifier
from dimensionality_reduction.train_autoencoder import AutoEncoderTraining


class XGBoostAutoencodersPredict(XGBoostPredict):
    batch_size = 64
    epochs = 100
    optimizer = SGD
    learning_rate = 0.01

    def __init__(self, definition, cutoff_point, features_interval,
                 prediction_start_day, prediction_interval, scale,
                 denoising):
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
            prediction_start_day: int
                the day where you want to make predictions for after the
                cutoff_point
            prediction_interval: int
                the interval where you will make predictions for
        """
        super().__init__(definition, cutoff_point, features_interval,
                         prediction_start_day, prediction_interval)

        self.scale = scale
        self.denoising = denoising

    def pipeline(self, X_train, y_train, X_test, y_test):
        """A method to pipeline the steps that need to be done in train and
        test. First oversamples the training set and then calculates features
        from the oversampled training set using autoencoders. Then applies the
        trained autoencoder to the test set. After that it
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
        autoencoder = AutoEncoderTraining(self.batch_size,
                                          torch.cuda.is_available(),
                                          self.scale, flatten=False)
        scalers, dataloader = autoencoder.preprocessing(X_train, mode='train')
        auto_model, X_train = autoencoder.train(
                self.epochs,
                self.optimizer,
                self.learning_rate,
                self.denoising,
                dataloader)

        xgboost_model = self.train(X_train, y_train)
        X_test = autoencoder.produce_encodings(auto_model, X_test, scalers)
        temp_scores = super().test(xgboost_model, X_test, y_test, scoring)

        return temp_scores


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
            "-sd",
            "--prediction_start_day",
            help="Choose the day you will start making predictions for",
            type=int,
            action="store",
            default=7
            )
    parser.add_argument(
            "-pi",
            "--prediction_interval",
            help="Choose the interval you are going to make predictions for",
            type=int,
            action="store",
            default=7
            )
    parser.add_argument(
            "-n",
            "--denoising",
            help="Choose if you are going to train the denoising version",
            action="store_true",
            default=True
            )
    parser.add_argument(
            "-s",
            "--scale",
            help="Choose if you are going to scale the features",
            action="store_true",
            default=False
            )
    args = parser.parse_args()
    scoring = {
            'auroc': roc_auc_score,
            'accuracy': accuracy_score,
            'f1': functools.partial(f1_score, average='macro')
            }
    test = XGBoostAutoencodersPredict(
            definition=args.definition,
            cutoff_point=args.cutoff_point,
            features_interval=args.features_interval,
            prediction_start_day=args.prediction_start_day,
            prediction_interval=args.prediction_interval,
            scale=args.scale,
            denoising=args.denoising
            )
    if args.data_type == 'sim':
        data, labels = test.get_data_and_labels(args.simulated_path)
        if args.mode == 'TT':
            data = test._stack_variables(data)
            X_train, X_test, y_train, y_test = train_test_split(
                            data, labels, test_size=0.2,
                            stratify=labels)

            scores = test.pipeline(X_train, y_train, X_test, y_test)
            test.write_to_csv(Classifier.xgboost_auto, DataType.simulated,
                              scores)
        else:
            scores = test.evaluate_simulated(data, labels, scoring)
            test.write_to_csv(Classifier.xgboost_auto, DataType.simulated,
                              scores)
    else:
        real_data, real_labels = test.get_data_and_labels(args.real_path)
        sim_data, sim_labels = test.get_data_and_labels(args.simulated_path)
        real_data = test._stack_variables(real_data)
        sim_data = test._stack_variables(sim_data)
        scores = test.pipeline(sim_data, sim_labels, real_data, real_labels)
        test.write_to_csv(Classifier.xgboost_auto, DataType.real, scores)
