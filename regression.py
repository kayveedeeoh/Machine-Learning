import numpy as np
import pandas as pd
import quandl
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model.base import LinearRegression

# Some global inputs to play with
forecast_col = 'Adj. Close'
percent_forecast = 0.01

# data frame: google stock ticker
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]

# we want to use the data to generate features for machine learning.
# useful features are Hi-Lo Percent, percent change, close and volume
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# our tailored dataset
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# it's wise not to have NaN's
df.fillna(-99999, inplace=True)

# look ahead at 1% of the length of the dataset
forecast_out = int(math.ceil(percent_forecast * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# features: everything except label column
X = np.array(df.drop(['label'], 1))

# labels
y = np.array(df['label'])

# skip if doing high-frequency trading.
# all data results need to be scaled too
X = preprocessing.scale(X)
df.dropna(inplace=True)

y = np.array(df['label'])

"""
This next section jumbles the rows, but keeps the relationship between X and y.
This is so that we can train the linearRegression model, and then test it on different
data so that we know that it is now able to get the answers right!
"""
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Create and train a classifier
clf = LinearRegression(n_jobs=-1)
# training data
clf.fit(X_train, y_train)
# test the data
accuracy = clf.score(X_test, y_test)
print(accuracy)
