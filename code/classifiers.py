from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dataset import DatapointKey as DK

import matplotlib.pyplot as plt
import numpy as np

from feature_extractor import FeatureExtractor

data = h5py.File("../data/labeled_output/data_preprocessed_labeled.h5", 'r')

extractor = FeatureExtractor(data)
n_bins = 10
variable = DK.TEMP_60_70
target = DK.CP07

variables = [DK.TEMP_60_70, DK.TEMP_80_90, DK.TEMP_60_90, DK.WIND_60, DK.WIND_65]

for variable in variables:
    X = extractor.hist(variable, n_bins=10)
    y = extractor.yearly_label(target)

    classifiers = [
        RandomForestClassifier(n_estimators=20),
        SVC(),
        LogisticRegression(),
        KNeighborsClassifier(n_neighbors=20),
        GaussianNB(),
        AdaBoostClassifier()
        ]

    scores = [cross_val_score(clf, X, y, cv=5, scoring=make_scorer(f1_score)) for clf in classifiers]

    clf_means = [np.mean(score) for score in scores]
    clf_2std = [2*np.std(score) for score in scores]

    classifiers_txt = [type(clf).__name__ for clf in classifiers]
    plt.figure()
    plt.errorbar(classifiers_txt, clf_means, yerr=clf_2std, marker=".",)
    plt.title("Histogram Features (n_bins={}). \nVariable: {}. Target: {}\n 5-fold CV".format(n_bins, variable, target))
    plt.savefig("{}.png".format(variable))
    plt.ylim(0, 1)


