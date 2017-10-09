from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
cancer = load_breast_cancer(return_X_y=True)

x= cancer[0]
y= cancer[1]

X_train , X_test, Y_train, Y_test = train_test_split(x,y)   # To Check this

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)                         # Transforming using
X_test = scaler.transform(X_test)                           # Feature Scaling

mlp = MLPClassifier(hidden_layer_sizes=(2,2,2))
mlp.fit(X_train,Y_train)

predictions = mlp.predict(X_train)
print(predictions)
count=0
for i in range(len(predictions)):
    predictions[i]= predictions[i]-Y_train[i]
    if(predictions[i]!=0):
        count=count+1;
    
print(count)
