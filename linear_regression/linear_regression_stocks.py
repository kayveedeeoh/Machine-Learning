import numpy as np
import pandas as pd
import quandl
import math
import datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model.base import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

# TODO: There's currently an issue where the forecast isn't future data. it's the last bit of the dataset.

style.use('ggplot')

# Some global inputs to play with
forecast_col = 'Adj. Close'
percent_forecast = 0.01

# data frame/dataset: google stock ticker
df = quandl.get('WIKI/GOOGL')
print(df.tail())

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

# features: everything except label column
X = np.array(df.drop(['label'], 1))

# skip scaling if doing high-frequency trading(slow).
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)

# labels
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

# predict future <forecast_col> values
forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

# set up dates to use on the graph
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# plot it
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
