"""
Cricket Chirps Vs. Temperature

In the following data
X = chirps/sec for the striped ground cricket
Y = temperature in degrees Fahrenheit
Reference: The Song of Insects by Dr.G.W. Pierce, Harvard College Press
https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel("https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/excel/slr02.xls")

dataset.plot(x="X", y="Y", style='o')
plt.title("Cricket Chirps Vs. Temperature")
plt.xlabel("Chirps/sec")
plt.ylabel("Temperature in degrees Fahrenheit")
plt.show()