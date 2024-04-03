#write your code here
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt

#define sine function
x = np.arange(-100 * np.pi, 100 * np.pi).reshape(-1,1) / 100
y = np.sin(x)

#initialize mlp
model = Sequential()
model.add(Dense(50, activation='sigmoid', input_shape=(1, )))
model.add(Dense(30, activation='sigmoid') )
model.add(Dense(1))                                  

#train the mlp
nb_epoch = 500
batch = 10
model.compile(loss='mean_squared_error', optimizer='SGD')
history = model.fit(x, y, epochs=nb_epoch, batch_size=batch)

#test
predictions = model.predict(x)
plt.plot(x, predictions)
plt.plot(x, np.sin(x))
plt.legend(['prediction', 'sin'])
plt.show()


