import pandas as pd
import numpy as np
from datetime import datetime
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras import optimizers

# read in data
def parse(x):
    return datetime.strptime(x, '%m/%d/%Y')

spxData = pd.read_csv("SPX.csv", parse_dates=["Date"], date_parser=parse)

# Determine green/red dates
spxData["green"] = spxData["Adj Close"].shift(1) <= spxData["Adj Close"]
# 200d sma
spxData["200sma"] = spxData["Adj Close"].rolling(window=200).mean()
# 50d sma
spxData["50sma"] = spxData["Adj Close"].rolling(window=50).mean()
spxData["50smavol"] = spxData["Volume"].rolling(window=50).mean()


spxData['close5perchange'] = spxData['Adj Close'].pct_change(5)
spxData['close10perchange'] = spxData['Adj Close'].pct_change(10)
spxData['close-1perchange'] = (spxData['Adj Close'].shift(-1) - spxData['Adj Close'])/spxData['Adj Close']
spxData.loc[spxData['close-1perchange']>0.01,'close-1perchange']=1
spxData.loc[spxData['close-1perchange']<=0.01,'close-1perchange']=0

# computed outputs
spxData["shouldbuy"] = spxData["Adj Close"].shift(1) >= spxData["Adj Close"]
spxData["gain"] = 100.0 * (spxData["Adj Close"].shift(1) - spxData["Adj Close"])/spxData["Adj Close"]
spxData["predictedClose"] = spxData["Adj Close"].shift(35)

# remove columns with na
spxData.dropna(inplace=True)

print(spxData.head(5))

# def normalize(df, col):
#     df[col] = df[col] / df[col].max()

# normalize(spxData, "Adj Close")
# normalize(spxData, "Volume")
# normalize(spxData, "50sma")
# normalize(spxData, "200sma")

# x_train = np.reshape(x_train, (1980, 5, 1))

# These are the columns we are interested in
input_cols = ["Adj Close", "Volume", "200sma", "50sma", "close5perchange", "50smavol","close10perchange"]

# Now iterate over the data frame, creating tuples for each of the set of inputs and then convert them to a list

input_data_initial = np.asarray(spxData[input_cols])
print(input_data_initial.shape)
output_data_initial = np.asarray(spxData['close-1perchange'])

# We build a sequential NN 
# input layer = num_features
# denselayer that accepts input and produces 30 outputs
# denselayer that accepts 30 outputs and produces output of size num_features
model = Sequential()
model.add(Dense(units=10, input_shape=(len(input_cols),), init="uniform",activation="tanh"))
#model.add(Dropout(0.1))
model.add(Dense(units=10, init="uniform",activation="tanh"))
#model.add(Dropout(0.1))
model.add(Dense(units=1, init="uniform",activation="sigmoid"))

# Stochastic gradient descent optimizer with some sensible defaults.
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.summary()

test_train_split = len(input_data_initial)-10
# Train the model.
model.fit(input_data_initial[:test_train_split], output_data_initial[:test_train_split], epochs=50, batch_size=300) # 18 seems to get to 1.0

h = model.evaluate(input_data_initial[test_train_split+1:], output_data_initial[test_train_split+1:])
print (h)
# # Create a random test vector of features to see how well we predict
# test_data = np.random.rand(1,len(input_cols)) * 10
# print (test_data)
# result = model.predict(np.asarray(test_data))
# print ("Predicted result: ",result)
# print ("   Actual result: ", test_data / 3)
# print ("-" * 20)
# print ("     Differences: ", (test_data / 3 ) - result)