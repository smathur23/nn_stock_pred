import pandas as pd
import numpy as np
from datetime import datetime
import pandas_ta as ta
import yfinance as yf
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
 
import warnings
warnings.filterwarnings("ignore")

ticker = input("Enter the ticker of stock to predict: ")
epochs = int(input("Enter # of epochs: "))
years = int(input("Enter # of years to train on: "))
stock = yf.Ticker(ticker)
current_date = date.today()
past_date = current_date - relativedelta(years=years)
data = stock.history(start=past_date, end=current_date, interval="1d")
data['Date'] = data.index

macd = data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)['MACD_12_26_9'].dropna()
data = data.drop(data.index[range(25)])
data['MACD'] = macd.values

close_data = data.filter(['Close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))

macd_data = data.filter(['MACD'])
macdfeatures = macd_data.values

from sklearn.preprocessing import MinMaxScaler
 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_macd = scaler.fit_transform(macdfeatures)

timesteps = 60

train_data = scaled_data[0:int(training), :] 
macd_train = scaled_macd[0:int(training), :]
x_train = []
y_train = []
macdfeature = []
macdfeature1 = []

for i in range(timesteps, len(train_data)):
    x_train.append(train_data[i-timesteps:i, :])
    y_train.append(train_data[i]) 
    macdfeature.append(macd_train[i-timesteps:i,0])
    macdfeature1.append(macd_train[i])

x_train, y_train, macdfeature = np.array(x_train), np.array(y_train), np.array(macdfeature)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
macdfeature = np.reshape(macdfeature, (macdfeature.shape[0], macdfeature.shape[1], 1))

import tensorflow as tf
from tensorflow import keras
from keras.layers import (Dropout, Flatten, Dense, Input, Concatenate, LSTM)
from keras.models import Model, Sequential

n = 1 
def quantile_loss(q, y, y_p):
        e = y-y_p
        return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

# quantile = float(input("Enter float value for quantile loss: "))

def network():
    sequence = Input(shape=(timesteps,1), name='Sequence')
    features = Input(shape=(timesteps,1), name='Features')

    lstm = Sequential()
    lstm.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, 1)))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=50, return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=50, return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=50))
    lstm.add(Dropout(0.2))

    part1 = lstm(sequence)

    flat_features = Flatten()(features)

    merged = Concatenate(axis=-1)([part1, flat_features])


    final = Dense(512, activation='relu')(merged)
    final = Dropout(0.5)(final)
    final = Dense(1)(final)

    model = Model(inputs=[sequence, features], outputs=[final])

    #model.compile(loss=lambda y, y_p: quantile_loss(quantile, y, y_p), optimizer = 'adam')
    model.compile(loss='mean_squared_error', optimizer = 'adam')

    return model

m = network()
m.fit([x_train, macdfeature], y_train, epochs=epochs, batch_size=32)

test_data = scaled_data[training-60:, :]
y_test = dataset[training:, :]

macd_test = scaled_macd[training-timesteps:, :]
x_test = []
macd_test_feature = []

for i in range(timesteps, len(test_data)):
    x_test.append(test_data[i-timesteps:i, 0])
    macd_test_feature.append(macd_test[i-timesteps:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

macd_test_feature = np.array(macd_test_feature)
macd_test_feature = np.reshape(macd_test_feature, (macd_test_feature.shape[0], macd_test_feature.shape[1], 1))

predictions = m.predict([x_test, macd_test_feature])

avgval = round(y_test.mean() / predictions.mean(), 2)
predictions = list(map(lambda x: x * avgval, predictions))

mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))


train = data[:training]
test = data[training:]
test['Predictions'] = predictions

import matplotlib.pyplot as plt
title = ticker + " Stock Close Price"
plt.figure(figsize=(10, 8))
plt.plot(train['Date'], train['Close'])
plt.plot(test['Date'], test[['Close', 'Predictions']])
plt.title(title)
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()