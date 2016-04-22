# John Elton jre2132
# Patrick Rogan psr2125
# Joshua Safyan jds2258
# Team 'JJ Reps'
# COMS 4721 Machine Learning
# Course Project

import scipy as sp
import numpy as np
import pandas as pd
import random
import math
import sys

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

##############################################################
# Data Ingestion / Processing
##############################################################
 
data_in = pd.read_csv(sys.argv[1])
test_in = pd.read_csv(sys.argv[2])
 
categorical_vars = [0, 5, 7, 8, 9, 14, 16, 17, 18, 20, 23, 25, 26, 56, 57, 58]
non_binary = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 59, 60]
categorical_vars = list(map(str, categorical_vars))
non_binary = list(map(str, non_binary))

def dummies_and_standardize(data, test):

    X = pd.get_dummies(data[data.columns[:-1]])
    test = pd.get_dummies(test)

	# get the columns in train that are not in test
    cols_to_add = np.setdiff1d(X.columns, test.columns)

	# add these columns to test, setting them equal to zero
    for col in cols_to_add:
        test[col] = 0

	# select and reorder the test columns using the train columns
    test = test[X.columns]

	#labels
    y = data[data.columns[-1]]
    return X, y, test

data, labels, quiz = dummies_and_standardize(data_in, test_in)

# scale non-binary numerical features to be between 0 and 1
data[non_binary] = data[non_binary].apply(lambda x: preprocessing.MinMaxScaler().fit_transform(x))
quiz[non_binary] = quiz[non_binary].apply(lambda x: preprocessing.MinMaxScaler().fit_transform(x))

# convert the data to sparse array format to reduce memory usage and increase speed
data_sparse = sp.sparse.csr_matrix(data, shape=None, dtype=None, copy=False)
quiz_sparse = sp.sparse.csr_matrix(quiz, shape=None, dtype=None, copy=False)
del data
del quiz

BoostingRF = AdaBoostClassifier(RandomForestClassifier(n_estimators=35, max_features='log2', n_jobs=-1, max_leaf_nodes = 9000, random_state = 100), n_estimators=10)
BoostingRF.fit(data_sparse, labels)
#scores_boosting = cross_validation.cross_val_score(BoostingRF, data_sparse, labels, cv=10)
preds1 = BoostingRF.predict(quiz_sparse)

#print(scores_boosting.mean())

submission = pd.DataFrame({"Prediction": preds1})
submission.index += 1
submission.to_csv(sys.argv[3], index_label="Id")




