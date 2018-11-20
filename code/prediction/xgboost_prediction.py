import argparse
import pickle
# import sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from xgboost_simple import ManualAndXGBoost
from sklearn.model_selection import train_test_split


class XGBoostPredict(ManualAndXGBoost):
    def __init__(self, definition, path, pickle_path, cutoff_point,
                 max_prediction):
        super().__init__(definition, path, pickle_path)
        self.cutoff_point = cutoff_point
        self.prediction_interval = list(range(5, max_prediction + 1, 5))

    def _get_labels_for_prediction(self):
        temp_labels = self.data_manager.get_data_for_variable(self.definition)
        labels = np.zeros((
                        temp_labels.shape[0], len(self.prediction_interval)))
        for i in range(len(temp_labels)):
            for j, offset in enumerate(self.prediction_interval):
                labels[i, j] = int(np.any(
                                   temp_labels[i, self.cutoff_point:
                                               self.cutoff_point + offset]))
        return labels

    def preprocess_as_prediction(self):
        labels = self._get_labels_for_prediction()
        # get distribution of the labels
        # for i in range(len(self.prediction_interval)):
        #     print(np.unique(labels[:, i], return_counts=True))
        try:
            with open(str(self.pickle_path), 'rb') as f:
                features, self.feature_keys = pickle.load(f)
            features = np.array(features)
        except FileNotFoundError:
            print("Didn't find the .pkl file of the features. Producing it",
                  "now, under the pickle path folder")
            data = super()._bring_data_to_format()
            data = data[:, :self.cutoff_point]
            features = self._produce_features(data)
            with open(str(self.pickle_path), 'wb') as f:
                pickle.dump([features, self.feature_keys], f)

        X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2,
                        stratify=labels[:, 0], random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        prediction_models = []
        for i, interval in enumerate(self.prediction_interval):
            prediction_models.append(super().train(X_train, y_train[:, i]))
        return prediction_models

    def test(self, prediction_models, X_test, y_test):
        for i, interval in enumerate(self.prediction_interval):
            y_pred = prediction_models[i].predict(X_test)
            auc = roc_auc_score(y_test[:, i], y_pred)
            f1 = f1_score(y_test[:, i], y_pred, average='macro')
            print(("{0} days in advance, \t AUROC: {1:.2f}, \t F1:"
                   "{2:.2f}").format(interval, auc, f1))


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
            "-o",
            "--output_path",
            help="Choose the output path of the pickle file",
            action="store",
            default="data/"
            )
    parser.add_argument(
            "-cp",
            "--cutoff_point",
            help="Choose the cutoff point of the time series",
            type=int,
            action="store",
            default=60
            )
    parser.add_argument(
            "-pi",
            "--prediction_interval",
            help="Choose the max prediction interval",
            type=int,
            action="store",
            default=30
            )
    args = parser.parse_args()
    pickle_path = (args.output_path + "features" + str(args.cutoff_point) +
                   ".pkl")
    test = XGBoostPredict(
            definition=args.definition,
            path=args.input_path,
            pickle_path=pickle_path,
            cutoff_point=args.cutoff_point,
            max_prediction=args.prediction_interval
            )
    X_train, X_test, y_train, y_test = test.preprocess_as_prediction()
    model = test.train(X_train, y_train)
    test.test(model, X_test, y_test)
