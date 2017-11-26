import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

# We assume we have 5 numbers of interest in our data.
num_features = 5

# Create 100000 lines of random data 
input = np.random.rand(100000, num_features)
# Our output is just the input with each feature increased by 5
output = np.divide(input, 3)

# We build a sequential NN with a fully connected layer
model = Sequential()
model.add(Dense(num_features, input_shape=(num_features,)))
#model.add(Dropout(0.1))
#model.add(Dense(num_features))

# Stochastic gradient descent optimizer with some sensible defaults.
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer="sgd",
              metrics=['accuracy'])

# Train the model.
model.fit(input, output, epochs=18) # 18 seems to get to 1.0

# Create a random test vector of features to see how well we predict
test_data = np.random.rand(1,num_features) * 10
print (test_data)
result = model.predict(np.asarray(test_data))
print (result)