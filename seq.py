import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import csv
import numpy as np
import matplotlib.pyplot as plt

class SEQ:
    def __init__(self, weights = "./seq", data_file="training_data.csv"):
        self.data_file = data_file
        self.weights = weights
        self.x_train = []
        self.y_train = []
        self.init_train()
        self.model = self.init_model()

    def init_train(self):
        d = dict()
        with open(self.data_file, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                tmp = []
                key = row[0] + ","+ row[1]
                if(key not in d):
                    d[key] = []
                for i in row:
                    tmp.append(float(i))
                row = tmp
                x = row[2:]
                d[key].append(x)

        for key in d:
            x = np.array(d[key])
            xmean = np.mean(x[:,0])
            ymean = np.mean(x[:,1])
            self.x_train.append([xmean, ymean])
            self.y_train.append([float(i) for i in key.split(',')])


    def init_model(self):
        model = Sequential()
        model = tf.keras.models.Sequential()  # a basic feed-forward model
        model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

        model.compile(optimizer='adam',  # Good default optimizer to start with
                      loss='mse',  # how will we calculate our "error." Neural network aims to minimize loss.
                      metrics=['accuracy'])  # what to track
        return model

    def load_weights(self):
        self.model.load_weights(self.weights)

    def save_weights(self):
        self.model.save_weights("weights")

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=100)

    def predict(self, x):
        self.model.predict(x)

x_train = []
y_train = []
with open('training_data.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        tmp = []
        for i in row:
            tmp.append(float(i))
        row = tmp
        x = np.array(row[2:])
        x_train.append(x)
        y = np.array([row[0], row[1]])
        y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)


model = Sequential()
model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='mse',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track
#
model.fit(x_train, y_train, epochs=100)  # train the model
# model.load_weights("./seq")
# val_loss, val_acc = model.evaluate(x_train, y_train)  # evaluate the out of sample data with model
# print(val_loss)  # model's loss (error)
# model.save_weights("./seq")


sample = x_train[0]
print("Prediction: ")
b = model.predict(x_train)
print(b)
print("Actual: ")
print(y_train[0])
