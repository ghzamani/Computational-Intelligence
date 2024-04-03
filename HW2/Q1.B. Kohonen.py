#write your code here

import numpy as np
# import matplotlib.pyplot as plt
import pylab as plt

class Kohonen:
  def __init__(self, input_neurons, map_dim, inputs, learning_rate, r, epoch):
    #neurons count
    self.input_neurons = input_neurons
    self.map_dim = map_dim
    self.learning_rate = learning_rate
    self.neighbor_radius = r 
    self.epoch = epoch
    #normalize the data
    self.inputs = np.divide(inputs, 255) 

    #initialize weights 
    self.weights = np.random.rand(self.map_dim[0], self.map_dim[1], 3)

#####################################################################################

  def decrease_lr(self, i):
      return self.learning_rate * np.exp(-i / self.epoch)

#####################################################################################

  def competition(self, data):
    repeated_arr = np.tile(data, (self.map_dim[0], self.map_dim[1], 1))
    sub = np.subtract(repeated_arr, self.weights)
    pow = np.power(sub, 2)
    sigma = np.sum(pow, axis=2)
    root = np.sqrt(sigma)
    index = np.unravel_index(np.argmin(root, axis=None), root.shape)
    return index
#####################################################################################
    
  #gausian function
  def cooperate(self, distances):
    return np.exp(((-distances) / (2 * self.neighbor_radius**2)))
#####################################################################################

  def changes(self, data, index, lr):
    h = self.cooperate(self.distToWinner(index))
    h = np.reshape(h, (h.shape[0], h.shape[1], 1))
    h = np.repeat(h, 3, axis=2)
    repeated = np.tile(data, (self.map_dim[0], self.map_dim[1], 1))
    change = np.multiply(lr , np.multiply(np.subtract(repeated, self.weights), h))
    self.weights += change
    
#####################################################################################
  def indices_array(self, m,n):
    r0 = np.arange(m) 
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out

  def distToWinner(self, index):
    winner_index = np.tile(np.array([[index[0], index[1]]]), (self.map_dim[0], self.map_dim[1], 1))
    indexes = self.indices_array(self.map_dim[0], self.map_dim[1])
    sub = np.subtract(winner_index, indexes)
    pow = np.power(sub, 2)
    sum = np.sum(pow, axis=2)
    return sum
#####################################################################################

  def learn(self):
    for e in range(self.epoch):
      lr = self.decrease_lr(e)
      for d in self.inputs:
        i = self.competition(d)
        self.changes(d, i, lr)
      title = "iteration " + str(e)
      plt.title(title)
      plt.imshow(self.weights)
      plt.show()

s = 40
epoch = 20
#array of rgb colors
rgb = np.random.randint(255, size=(s**2, 3))
plt.imshow(rgb.reshape(s, s, 3))
plt.title("input colors")
plt.show()

k_map = Kohonen(s**2, (s, s), rgb, 0.1, 5, epoch)
k_map.learn()


#write your code here

import numpy as np
# import matplotlib.pyplot as plt
import pylab as plt

class Kohonen:
  def __init__(self, input_neurons, map_dim, inputs, learning_rate, r, epoch):
    #neurons count
    self.input_neurons = input_neurons
    self.map_dim = map_dim
    self.learning_rate = learning_rate
    self.neighbor_radius = r 
    self.epoch = epoch
    #normalize the data
    self.inputs = np.divide(inputs, 255) 

    #initialize weights 
    self.weights = np.random.rand(self.map_dim[0], self.map_dim[1], 3)

    self.timeconst = self.epoch / np.log(self.neighbor_radius)

#####################################################################################
  def decrease_radius(self, i):
      return self.neighbor_radius * np.exp(-i / self.timeconst)

#####################################################################################

  def competition(self, data):
    repeated_arr = np.tile(data, (self.map_dim[0], self.map_dim[1], 1))
    sub = np.subtract(repeated_arr, self.weights)
    pow = np.power(sub, 2)
    sigma = np.sum(pow, axis=2)
    root = np.sqrt(sigma)
    index = np.unravel_index(np.argmin(root, axis=None), root.shape)
    return index
#####################################################################################
    
  #gausian function
  def cooperate(self, distances, r):
    return np.exp(((-distances) / (2 * r**2)))
#####################################################################################

  def changes(self, data, index, r):
    h = self.cooperate(self.distToWinner(index), r)
    h = np.reshape(h, (h.shape[0], h.shape[1], 1))
    h = np.repeat(h, 3, axis=2)
    repeated = np.tile(data, (self.map_dim[0], self.map_dim[1], 1))
    change = np.multiply(self.learning_rate , np.multiply(np.subtract(repeated, self.weights), h))
    self.weights += change
    
#####################################################################################
  def indices_array(self, m,n):
    r0 = np.arange(m) 
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out

  def distToWinner(self, index):
    winner_index = np.tile(np.array([[index[0], index[1]]]), (self.map_dim[0], self.map_dim[1], 1))
    indexes = self.indices_array(self.map_dim[0], self.map_dim[1])
    sub = np.subtract(winner_index, indexes)
    pow = np.power(sub, 2)
    sum = np.sum(pow, axis=2)
    return sum
#####################################################################################

  def learn(self):
    for e in range(self.epoch):
      r = self.decrease_radius(e)
      for d in self.inputs:
        i = self.competition(d)
        self.changes(d, i, r)
      title = "iteration " + str(e)
      plt.title(title)
      plt.imshow(self.weights)
      plt.show()

s = 40
epoch = 20
#array of rgb colors
rgb = np.random.randint(255, size=(s**2, 3))
plt.imshow(rgb.reshape(s, s, 3))
plt.title("input colors")
plt.show()

k_map = Kohonen(s**2, (s, s), rgb, 0.1, 5, epoch)
k_map.learn()


