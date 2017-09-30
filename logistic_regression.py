#iris flower prediction using logistic regression.
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

iris= load_iris()
x =iris.data[:-1]
y= iris.target[:-1]
logistic= LogisticRegression()
logistic.fit(x,y)

print('Predicted Probabilities are {}'.format(logistic.predict_proba(iris.data[-1,:])))
print('Predicted class is {}'.format(logistic.predict(iris.data[-1,:])))


