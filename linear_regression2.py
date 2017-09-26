#Implementing linear regression using scikit-learn.

from sklearn.datasets import load_boston
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
boston = load_boston()
df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame( boston.target)

regr= linear_model.LinearRegression()
x_train , x_test, y_train, y_test= train_test_split(df_x, df_y, test_size= 0.2, random_state= 4)

regr.fit(x_train, y_train)
plt.scatter(regr.predict(x_test), y_test)
plt.show()

