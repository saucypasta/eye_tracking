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
        with open(self.data_file, newline='') as csvfile:
            with open('training_data.csv', newline='') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',')
                for row in datareader:
                    tmp = []
                    for i in row:
                        tmp.append(float(i))
                    row = tmp
                    a = row[2:6] + row[-4:]
                    x = np.array(a)
                    self.x_train.append(x)
                    y = np.array([row[0], row[1]])
                    self.y_train.append(y)
            self.x_train = np.array(self.x_train)
            self.y_train = np.array(self.y_train)

    def init_model(self):
        model = Sequential()
        model.add(Dense(256,input_dim=8,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(2,activation='linear'))
        model.compile(loss='MSE')
        return model

    def load_weights(self):
        self.model.load_weights(self.weights)

    def save_weights(self):
        self.model.save_weights(self.weights)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=1000)

    def predict(self, x):
        p = self.model.predict(x)
        print(p)

seq = SEQ()
