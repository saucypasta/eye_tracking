import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import csv
import numpy as np


x_train = []
y_train = []
with open('training_data.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        tmp = []
        for i in row:
            tmp.append(float(i))
        row = tmp
        img = (np.array(row[6:])/255).tolist()
        x = row[0:6] + img
        x_train.append(x)
        y = np.array([row[0], row[1]])
        y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)


model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='mse',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=10)  # train the model

val_loss, val_acc = model.evaluate(x_train, y_train)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

sample = x_train[0]
print("Prediction: ")
a = model.predict(np.array([sample]))
print(a)
print("Actual: ")
print(y_train[0])
