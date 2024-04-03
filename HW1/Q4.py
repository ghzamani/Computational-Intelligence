# Q4_graded
# Do not change the above line.

# This cell is for your imports.

from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils 
import seaborn as sns
from keras.initializers import RandomNormal
import time

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert the 2d vector into 1d vector of 28^2
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])


# normalize the data
X_train = X_train/255
X_test = X_test/255

# encode output with one hot
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

output_dim = 10
input_dim = X_train.shape[1]
nb_epoch = 20

myMLP = Sequential()
myMLP.add(Dense(512, activation='relu', input_shape=(input_dim,)))
myMLP.add(Dense(128, activation='relu') )
myMLP.add(Dense(output_dim, activation='softmax'))

myMLP.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

history = myMLP.fit(X_train, Y_train, epochs=nb_epoch, validation_data=(X_test, Y_test))

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

loss = history.history['val_loss']
accuracy = history.history['val_accuracy']
ax.plot(x, loss, 'b', label="Validation Loss")
ax.plot(x, accuracy, 'r', label="Validation Accuracy")
plt.show()


