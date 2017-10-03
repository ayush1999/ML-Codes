from sklearn import svm
from sklearn.datasets import load_iris
iris = load_iris()
model = svm.SVC(kernel = 'linear', gamma=1)
x= iris.data[0:-1]
y = iris.target[0:-1]
model.fit(x,y)
predicted = model.predict(x[-1])
print('Preddicted value is {}'.format(predicted))