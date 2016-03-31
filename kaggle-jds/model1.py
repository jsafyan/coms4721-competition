import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


#############################
# Data Ingestion / Processing
#############################

data = pd.read_csv("data.csv")
test = pd.read_csv("quiz.csv")


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

X, y, test = dummies_and_standardize(data, test)

#scale non-binary numerical features to be between 0 and 1
X[non_binary] = X[non_binary].apply(lambda x: preprocessing.MinMaxScaler().fit_transform(x))
test[non_binary] = test[non_binary].apply(lambda x: preprocessing.MinMaxScaler().fit_transform(x))

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
model2.score(X_test, y_test)

#stochastic gradient descent model--using hinge loss -> SVC, 'log' loss gives logistic classifier
clf = linear_model.SGDClassifier(loss="hinge", n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
clf.fit(X, y)
clf.score(X, y)

def create_submission_csv(clf, test):
	submission = pd.DataFrame({"Prediction": clf.predict(test)})
	submission.index += 1
	submission.to_csv("submission.csv", index_label="Id")

create_submission_csv(clf, test)
