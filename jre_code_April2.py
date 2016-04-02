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
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

sp.random.seed(1)

TRAINING_RUN = 1

##############################################################
# Data Ingestion / Processing
##############################################################
 
data_in = pd.read_csv("data.csv")
test_in = pd.read_csv("quiz.csv")
 
 
categorical_vars = [0, 5, 7, 8, 9, 14, 16, 17, 18, 20, 23, 25, 26, 56, 57, 58]
non_binary = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 59, 60]
categorical_vars = list(map(str, categorical_vars))
non_binary = list(map(str, non_binary))

def dummies_and_standardize(data, test):
	# see: http://stackoverflow.com/questions/34170413/possible-ways-to-do-one-hot-encoding-in-scikit-learn
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

#scale non-binary numerical features to be between 0 and 1
data[non_binary] = data[non_binary].apply(lambda x: preprocessing.MinMaxScaler().fit_transform(x))
quiz[non_binary] = quiz[non_binary].apply(lambda x: preprocessing.MinMaxScaler().fit_transform(x))

# Get N and dim values for data and quiz
quiz = quiz.values
N = data.shape[0]
dim = data.shape[1] 
N_quiz = quiz.shape[0]
dim_quiz = quiz.shape[1]

# convert the data to sparse array format to reduce memory usage and increase speed
data_sparse = sp.sparse.csr_matrix(data, shape=None, dtype=None, copy=False)
quiz_sparse = sp.sparse.csr_matrix(quiz, shape=None, dtype=None, copy=False)
del data
del quiz

# use SelectKBest to reduce dimension of data
KBest = SelectKBest(f_classif, k=200)
data_best = KBest.fit_transform(data_sparse, labels)
ind_best = KBest.get_support(indices=True)
quiz_best = quiz_sparse[:,ind_best]

# Some examples to test algorithms on dimension-reduced data with cross validation

Boosting = AdaBoostClassifier(n_estimators=100)
scores_boosting = cross_validation.cross_val_score(Boosting, data_best, labels, cv=5)
print scores_boosting

RandomForest = RandomForestClassifier(n_estimators=10, n_jobs=-1)
RandomForest.fit(data_best, labels)
scores_forest = cross_validation.cross_val_score(RandomForest, data_best, labels, cv=5)
print scores_forest

#preds = RandomForest.predict(quiz_best)
#submission = pd.DataFrame({"Prediction": preds})
#submission.index += 1
#submission.to_csv("submission1.csv", index_label="Id")

Logistic = LogisticRegression(C=1.0)
scores_logistic = cross_validation.cross_val_score(Logistic, data_best, labels, cv=5)
print scores_logistic
