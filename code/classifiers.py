from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from dataset import DatapointKey as DK

import matplotlib.pyplot as plt
import numpy as np

from feature_extractor import FeatureExtractor

data_file = "../data/labeled_output/data_preprocessed_labeled.h5"

extractor = FeatureExtractor(data_file)
n_bins = 20

target = DK.CP07

variables = [DK.TEMP_60_70, DK.TEMP_80_90, DK.TEMP_60_90, DK.WIND_60, DK.WIND_65]

#for variable in variables:

variable = "TEMP_60_90 + WIND"
X_temp = extractor.hist(DK.TEMP_60_90, n_bins=n_bins)
X_wind60 = extractor.hist(DK.WIND_60, n_bins=n_bins)
X_wind65 = extractor.hist(DK.WIND_65, n_bins=n_bins)

X = np.concatenate([X_temp, X_wind60, X_wind65], axis=1)

y = extractor.yearly_label(target)

classifiers = [
    RandomForestClassifier(n_estimators=100),
    AdaBoostClassifier(),
    XGBClassifier(n_estimators=100),
    #SVC(),
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=20),
    GaussianNB()
    ]

scores = [cross_val_score(clf, X, y, cv=5, scoring=make_scorer(f1_score)) for clf in classifiers]

clf_means = [np.mean(score) for score in scores]
clf_2std = [2*np.std(score) for score in scores]

classifiers_txt = [type(clf).__name__ for clf in classifiers]
plt.figure()
plt.bar(classifiers_txt, clf_means, yerr=clf_2std, align='center', alpha=0.5, ecolor='black', capsize=10)
# plt.errorbar(classifiers_txt, clf_means, yerr=clf_2std, marker=".",)
plt.title("Binary Classification Using Histogram Features \n Variables: {}. Target: {}\n 5-fold CV".format(variable, target))
plt.savefig("{}.png".format(variable))
plt.ylim(.7, 1)

plt.show()

