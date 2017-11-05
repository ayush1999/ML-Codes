from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
iris = load_iris()
x = iris.data
y = iris.target

mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100))
mlp.fit(x, y)

predictions = mlp.predict(x)

print(predictions)

for i in range(len(predictions)):
    predictions[i] = predictions[i] - y[i]

print(predictions)
