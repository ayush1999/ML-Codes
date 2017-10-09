from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = np.loadtxt('data.csv',delimiter=',')
x = dataset[:,0:8]                                                  # Classify into features and targets
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8 , activation='relu'))               # units in first layer
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(12, activation='relu'))
# units in next layer
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x,y,epochs=1000, batch_size=10)

scores = model.evaluate(x,y)
print('{} and {}'.format(model.metrics_names[1],scores[1]*100))

predictions = model.predict(x)
rounded = [round(x[0]) for x in predictions]
count = 0
for i in range(len(rounded)):
    if rounded[i]!= y[i]:
        count= count+1
print(rounded)
print('Number of incorrect answers are {}'.format(count))
