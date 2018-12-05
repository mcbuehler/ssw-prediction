from tsfresh import extract_features
import pandas as pd
import numpy as np


class FeatureEngineering:
    """A class that does feature engineering by using the tsfresh package. It
    also extract the feature keys in order to be able to see the importances of
    the XGBoost classifier."""

    def _construct_feature_keys(self, column_values, variables):
        """Get the column values of one of the variables and since they are the
        same for all the variables because of tsfresh it expands the features
        by prepending the name of the variable.
        Parameters
        ----------
            column_values: list
                A list with the values of the keys returned by tsfresh

        Returns
        -------
            feature_keys: list
                A list with the values of the keys for all the variables of
                length len(column_values)*len(self.variables)
        """
        temp_feature_keys = column_values
        keys_features_length = len(temp_feature_keys)
        variables_length = len(variables)
        feature_keys = []
        for i in range(keys_features_length*variables_length):
            quotient = int(i / keys_features_length)
            feature_keys.append(
                self.variables[quotient] + "_" +
                temp_feature_keys[
                     i-quotient*keys_features_length]
                 )

        return feature_keys

    def _produce_features(self, data, variables):
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
            if flag:
                flag = False
                feature_keys = self._construct_feature_keys(
                        X.columns.values, variables)

            features.append(X.values[0])

        # bring the features from format [num_data*num_features, len_winter] to
        # [num_data, num_features*len_winter]
        length = len(variables)
        new_features = [
                np.concatenate((features[i], features[i-1],
                                features[i-2]), axis=0)
                for i, feature in enumerate(features)
                if i % length == 2
                ]

        features = new_features[:]
        return feature_keys, np.asarray(features)
