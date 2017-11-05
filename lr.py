import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import random


def f(x):
	res = x*25 + 3
	error = random.uniform(-0.01, 0.01)
	return res+error


values = []
for i in range(0, 300):
	x = random.uniform(0, 1000)
	y = f(x)
	values.append([[x], y])

regr = LinearRegression()
x, y = zip(*values)
x_train = list(x[:-20])
y_train = list(y[:-20])
x_test = list(x[-20:])
y_test = list(y[-20:])

regr.fit(x_train, y_train)
print(regr.coef_)
print(regr.intercept_)
