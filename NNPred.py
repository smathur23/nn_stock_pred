import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta

ticker = input("Enter the ticker of stock to predict: ")
epochs = int(input("Enter # of epochs: "))
years = int(input("Enter # of years to train on: "))
stock = yf.Ticker(ticker)
current_date = date.today()
past_date = current_date - relativedelta(years=years)
df = stock.history(start=past_date, end=current_date, interval="1d")

print(df.head())
df['Date'] = df.index

# df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])

#df = df.drop(['Adj Close'], axis=1)

macd_result = df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
df = df.dropna(subset=['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'])
df.rename(columns={'MACD_12_26_9': 'MACD', 'MACDs_12_26_9': 'MACDs', 'MACDh_12_26_9': 'MACDh'}, inplace=True)

close_data = df.filter(['Close'])
dataset = close_data.values

macd_data = df.filter(['MACD', 'MACDh', 'MACDs'])
macdfeatures = macd_data.values

steps = 60

training = len(dataset) - 300
testing = len(dataset) - steps

print((training, testing))

from sklearn.preprocessing import MinMaxScaler
 
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)
scaled_macd = scaler.fit_transform(macdfeatures)

train_scaled = scaled_data[:training]
train_macd = scaled_macd[:training]

pred_scaled = scaled_data[training:testing]
pred_macd = scaled_macd[training:testing]

test = dataset[testing:]

other_labels = dataset[steps+1:training+1]

train_window = []
macd_wind = []
for i in range(steps, training):
    train_window.append(train_scaled[i-steps:i, :])
    macd_wind.append(train_macd[i-steps:i, :])

pred_window = []
pred_macd_wind = []
for i in range(steps, testing - training):
    pred_window.append(pred_scaled[i-steps:i, :])
    pred_macd_wind.append(pred_macd[i-steps:i, :])

train_window, macd_wind, pred_macd, pred_scaled, other_labels, pred_window, pred_macd_wind = np.array(train_window), np.array(macd_wind), np.array(pred_macd), np.array(pred_scaled), np.array(other_labels), np.array(pred_window), np.array(pred_macd_wind)

print(macd_wind.shape)
print(train_window.shape)
# print(labels.shape)
print(other_labels.shape)

train_window = np.reshape(train_window, (train_window.shape[0], train_window.shape[1], 1))
macd_wind = np.reshape(macd_wind, (macd_wind.shape[0], macd_wind.shape[1], 3))

pred_window = np.reshape(pred_window, (pred_window.shape[0], pred_window.shape[1], 1))
pred_macd_wind = np.reshape(pred_macd_wind, (pred_macd_wind.shape[0], pred_macd_wind.shape[1], 3))

import tensorflow as tf 
from tensorflow import keras

model = keras.models.Sequential() 
model.add(keras.layers.LSTM(units=64, 
                            return_sequences=True, 
                            input_shape=(train_window.shape[1], 1))) 
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dropout(0.5)) 
model.add(keras.layers.Dense(1))


model.compile(optimizer='adam', 
              loss='mean_squared_error') 
history = model.fit([train_window, macd_wind], 
                    other_labels, 
                    epochs=epochs) 

print(pred_macd_wind.shape)
print(pred_window.shape)
print(test.shape)

predictions = model.predict([pred_window, pred_macd_wind])
diff = dataset[testing-1] - predictions[0]
predictions = [x + diff for x in predictions]

future = predictions[60:]

pred1 = predictions[:60]

try:
    mse = np.mean(((predictions - test) ** 2))
    print('Good')
except:
    mse = np.mean(((pred1 - test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

train = df[:testing]
test = df[testing:]
test['Predictions_test'] = pred1

import matplotlib.pyplot as plt
title = ticker + " Stock Close Price"
plt.figure(figsize=(10, 8))
plt.plot(train['Date'], train['Close'])
plt.plot(test['Date'], test[['Close', 'Predictions_test']])
plt.title(title)
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()
