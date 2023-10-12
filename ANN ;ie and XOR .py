
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y_AND = np.array([0, 0, 0, 1])
Y_OR = np.array([0, 1, 1, 1])
Y_XOR = np.array([0, 1, 1, 0])

model = Sequential()


model.add(Dense(1, input_dim=2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, Y_AND, epochs=10000, verbose=0)

print("AND Gate Predictions:")
print(model.predict(X).round())

model = Sequential()

model.add(Dense(1, input_dim=2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, Y_OR, epochs=10000, verbose=0)

print("\nOR Gate Predictions:")
print(model.predict(X).round())

model = Sequential()

model.add(Dense(1, input_dim=2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, Y_XOR, epochs=10000, verbose=0)

print("\nXOR Gate Predictions:")
print(model.predict(X).round())




     
