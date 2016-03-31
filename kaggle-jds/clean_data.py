import pandas as pd
import numpy as np
import pandas as pd

from sklearn import preprocessing

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

X.to_csv("training_features.csv", index=False)
y.to_csv("training_labels.csv", index=False)
test.to_csv("test_cleaned.csv", index=False)