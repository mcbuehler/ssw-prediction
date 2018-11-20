import argparse
import pickle
# import sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from classification.xgboost_simple import ManualAndXGBoost
from prediction_set import PredictionSet
from sklearn.model_selection import train_test_split


class XGBoostPredict(ManualAndXGBoost):
    def __init__(self, definition, path, pickle_path, cutoff_point,
                 prediction_interval):
        self.prediction_interval = prediction_interval
        self.pickle_path = pickle_path
        self.prediction_set = PredictionSet(definition, path, cutoff_point,
                                            prediction_interval)

    def _bring_data_to_format(self, temp_data):
        feature_count = temp_data.shape[1]
        data = []
        for i in range(len(temp_data)):
            for j in range(feature_count):
                data.append(temp_data[i, j, :])
        data = np.array(data)
        return data

    def preprocess_as_prediction(self):
        labels = np.ravel(self.prediction_set.get_labels_for_prediction())
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
            data = self.prediction_set.cutoff_for_prediction()
            data = self._bring_data_to_format(data)
            features = self._produce_features(data)
            with open(str(self.pickle_path), 'wb') as f:
                pickle.dump([features, self.feature_keys], f)

        X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2,
                        stratify=labels, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        model = super().train(X_train, y_train)
        return model

    def test(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(("{0} days in advance, \t AUROC: {1:.2f}, \t F1:"
               "{2:.2f}").format(self.prediction_interval, auc, f1))


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
            default=5
            )
    args = parser.parse_args()
    pickle_path = (args.output_path + "features" + str(args.cutoff_point) +
                   ".pkl")
    test = XGBoostPredict(
            definition=args.definition,
            path=args.input_path,
            pickle_path=pickle_path,
            cutoff_point=args.cutoff_point,
            prediction_interval=args.prediction_interval
            )
    X_train, X_test, y_train, y_test = test.preprocess_as_prediction()
    model = test.train(X_train, y_train)
    test.test(model, X_test, y_test)
