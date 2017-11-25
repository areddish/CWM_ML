from keras import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
from datetime import datetime

def parse(x):
	return datetime.strptime(x, '%m/%d/%Y')
# read in data
spxData = pd.read_csv("SPX.csv", parse_dates=["Date"], date_parser=parse)

# 200d sma
spxData["Adj Close"].rolling(window=200).mean()
# 50d sma
spxData["Adj Close"].rolling(window=50).mean()

#df["50vol"] = df["Volume"].rolling(window=50).mean()

spxData["green"] = spxData["Adj Close"].shift(1) <= spxData["Adj Close"]
spxData["200sma"] = spxData["Adj Close"].rolling(window=50).mean()
spxData["50sma"] = spxData["Adj Close"].rolling(window=50).mean()
print(spxData.head())
spxData.dropna(inplace=True)
print(spxData.head())

# input shape
# batch, x-axis, y-axis (x-axis = time, y axis = indicators

x_train = spxData[["Adj Close", "Volume", "green", "200sma", "50sma"]]
y_train = spxData["green"]

print("x_train: ",x_train.shape)
print("       : ",x_train.shape[1:])

x_train = np.reshape(x_train, (1980, 5, 1))

model = Sequential();
model.add(LSTM(input_shape=x_train.shape[1:], #(1980, 5)
   return_sequences=False,
   units=8))
#model.add(Dense)
#model.add(Dropout(0.1))
#model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

model.fit(
    x_train.values,
    y_train.values,
    epochs=5,
    batch_size=20)

   # validation_split=0.1,
   # verbose=1,
   # shuffle=True) #?
