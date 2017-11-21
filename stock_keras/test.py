from keras import Sequential
from keras.layers import Dense, LSTM
import pandas as pd

# read in data
spxData = pd.read_csv("SPX.csv")

# 200d sma
spxData["Adj Close"].rolling(window=200).mean()
# 50d sma
spxData["Adj Close"].rolling(window=50).mean()

#df["50vol"] = df["Volume"].rolling(window=50).mean()

spxData["green"] = spxData["Adj Close"].shift(1) <= spxData["Adj Close"]
spxData["200sma"] = spxData["Adj Close"].rolling(window=50).mean()
spxData["50sma"] = spxData["Adj Close"].rolling(window=50).mean()

model = Sequential();
model.add(LSTM(input_shape=(4,3),
   output_dim=1,
   return_sequences=False))
#model.add(Dense)
#model.add(Dropout(0.1))
#model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

model.fit(
    x_train,
    y_train,
    batch_size=50,
    nb_epoch=5,
    validation_split=0.1,
    verbose = 0,
    shuffle=True) #?
