import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import csv
import numpy as np


class RNN:
    def __init__(self, data_file = 'training_data.csv', weights="./rnn_weights"):
        self.data_file = data_file
        self.weights = weights
        self.x_train = []
        self.y_train = []
        self.init_train()
        self.model = self.init_model()

    def init_train(self):
        with open(self.data_file, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                tmp = []
                for i in row:
                    tmp.append(float(i))
                row = tmp
                x = np.array(row[6:])
                x = np.reshape(x, (50, 50))/255
                self.x_train.append(x)
                y = np.array([row[0], row[1]])
                self.y_train.append(y)
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

    def init_model(self):
        model = Sequential()

        # IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
        model.add(LSTM(2500, input_shape=(self.x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(1000))
        model.add(Dropout(0.1))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='relu'))

        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        # Compile model
        model.compile(
            loss='mse',
            optimizer=opt,
            metrics=['accuracy'],
        )
        return model

    def train(self):
        self.model.fit(self.x_train,
                  self.y_train,
                  epochs=3,
                  validation_data=(self.x_train, self.y_train))

    def save(self):
        self.model.save_weights("./numbers")

r = RNN()
r.train()
sample = r.x_train[0]
y = r.model.predict(sample)
print("prediction ", y)
print("actual ", r.y_train[0])

sample = r.x_train[-1]
y = r.model.predict(sample)
print("prediction ", y)
print("actual ", r.y_train[-1])
