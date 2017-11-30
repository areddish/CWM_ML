import pandas as pd
import numpy as np
from datetime import datetime

# read in data
def parse(x):
    return datetime.strptime(x, '%m/%d/%Y')

spxData = pd.read_csv("SPX.csv", parse_dates=["Date"], date_parser=parse)

#df["50vol"] = df["Volume"].rolling(window=50).mean()

spxData["green"] = spxData["Adj Close"].shift(1) <= spxData["Adj Close"]
# 200d sma
spxData["200sma"] = spxData["Adj Close"].rolling(window=200).mean()
# 50d sma
spxData["50sma"] = spxData["Adj Close"].rolling(window=50).mean()

spxData["shouldbuy"] = spxData["Adj Close"].shift(1) >= spxData["Adj Close"]
spxData["gain"] = 100.0 * (spxData["Adj Close"].shift(1) - spxData["Adj Close"])/spxData["Adj Close"]
spxData.dropna(inplace=True)

print(spxData.head(5))

def normalize(df, col):
    df[col] = df[col] / df[col].max()

normalize(spxData, "Adj Close")
normalize(spxData, "Volume")
normalize(spxData, "50sma")
normalize(spxData, "200sma")

# x_train = np.reshape(x_train, (1980, 5, 1))

# These are the columns we are interested in
input_cols = ["Adj Close", "Volume", "green", "200sma", "50sma"]

# Now iterate over the data frame, creating tuples for each of the set of inputs and then convert them to a list

input_data_initial = np.asarray(spxData[input_cols].apply(tuple, axis=1).apply(list))
print (input_data_initial[:3])
input_data = np.hstack(input_data_initial).reshape(len(spxData),1,len(input_cols))
print (np.hstack(input_data_initial[:3]))

output_data_initial = np.asarray(spxData[["shouldbuy"]].apply(tuple, axis=1).apply(list))
output_data = np.hstack(output_data_initial).reshape(len(spxData),1,1)

# Put your inputs into a single list
# Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
#spxData['single_input_vector'] = spxData.single_input_vector.apply(lambda x: [list(x)])
# Use .cumsum() to include previous row vectors in the current row list of vectors
#spxData['cumulative_input_vectors'] = spxData.single_input_vector.cumsum()

 #Get your input dimensions
# Input length is the length for one input sequence (i.e. the number of rows for your sample)
# Input dim is the number of dimensions in one input vector (i.e. number of input columns)
input_length = input_data.shape[1]
input_dim = input_data.shape[2]

# Output dimensions is the shape of a single output vector
# In this case it's just 1, but it could be more
output_dim = len(output_data[0])

from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras import optimizers

model = Sequential()
model.add(LSTM(input_shape=input_data.shape[1:],
   return_sequences=True,
   units=50))
model.add(Dropout(0.05))
model.add(LSTM(input_shape=input_data.shape[1:],
   return_sequences=True,
   units=200))
model.add(Dropout(0.05))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer="sgd",
              metrics=['accuracy'])
#model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])

model2 = Sequential()
model2.add(LSTM(input_shape=input_data.shape[1:],
   return_sequences=True,
   units=32))
model2.add(LSTM(32,return_sequences=True))
model2.add(LSTM(32,return_sequences=True))
model2.add(Dense(1, activation="softmax"))

model2.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


test_split_index = int(len(input_data) * 0.9)
test_input = input_data[test_split_index:]
test_output = output_data[test_split_index:]

model.fit(
    input_data[:test_split_index],
    output_data[:test_split_index],
    shuffle=True,
    epochs=5,
    batch_size=45)

model2.fit(
    input_data[:test_split_index],
    output_data[:test_split_index],
    epochs=5,
    batch_size=25)

score = model.evaluate(test_input, test_output, batch_size=25)
print("SCORE:", model.metrics_names, score)

#print("PREDICT: ", model.predict(test_input))





spxData['close5perchange'] = spxData['Adj Close'].pct_change(5)
spxData['close10perchange'] = spxData['Adj Close'].pct_change(10)
spxData['close-1perchange'] = (spxData['Adj Close'].shift(-1) - spxData['Adj Close'])/spxData['Adj Close']
spxData.loc[spxData['close-1perchange']>0.01,'close-1perchange']=1
spxData.loc[spxData['close-1perchange']<=0.01,'close-1perchange']=0
# def normalize(df, col):
#     df[col] = df[col] / df[col].max()

# normalize(spxData, "Adj Close")
# normalize(spxData, "Volume")
# normalize(spxData, "50sma")
# normalize(spxData, "200sma")

