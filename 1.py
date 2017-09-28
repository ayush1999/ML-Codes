import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split as tts
boston = datasets.load_boston()

x,y = np.arange(10).reshape(5,2), range(5)

x_train, x_test, y_train, y_test= tts(x, y, test_size=2 ,random_state= 2)
print(x_train)
print(y_train)