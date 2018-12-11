"""
Cricket Chirps Vs. Temperature

In the following data
X = chirps/sec for the striped ground cricket
Y = temperature in degrees Fahrenheit
Reference: The Song of Insects by Dr.G.W. Pierce, Harvard College Press
https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html

The function of the program is to predict how many cricket chirps/sec will result from a given temperature."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel("https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/"
                        "slr/excel/slr02.xls")

X = dataset.iloc[:, 0].values.reshape(-1, 1)  # Attributes
y = dataset.iloc[:, 1].values.reshape(-1, 1)  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

m = regressor.coef_[0]
b = regressor.intercept_

stop = dataset.max()[0]
start = dataset.min()[0]

y_prediction = regressor.predict(X_test)
y_prediction = [y for x in y_prediction for y in x]
X_test = [y for x in X_test for y in x]

df = pd.DataFrame({'Test': list(X_test), 'Prediction': list(y_prediction)})

# Plot the data
ax = dataset.plot(x="X", y="Y", style='o')
x = np.linspace(start, stop)
plt.plot(x, m * x + b)
df.plot(x='Test', y='Prediction', style='r+', ax=ax)

plt.title("Cricket Chirps Vs. Temperature")
plt.xlabel("Chirps/sec")
plt.ylabel("Temperature in degrees Fahrenheit")
plt.show()
