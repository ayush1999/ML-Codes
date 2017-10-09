from sklearn import svm
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
model = svm.SVC(gamma=1)
x= iris.data[0:-1]
y = iris.target[0:-1]
model.fit(x,y)
predicted = model.predict(x[-1])
print('Predicted value is {}'.format(predicted))