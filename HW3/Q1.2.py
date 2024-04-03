# Q1.2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1.2_graded
# Do not change the above line.

# This cell is for your codes.

class Hopfield:
  def __init__(self, patterns):
    self.length = patterns.shape[1]
    self.weights = np.zeros((self.length, self.length))
    self.patterns = patterns
    self.calculate_weights()

  def calculate_weights(self):
    for i in range(self.patterns.shape[0]):
      for j in range(self.length):
        for k in range(j+1, self.length):
          x = self.patterns[i, j] * self.patterns[i, k]
          self.weights[j, k] += x
          self.weights[k, j] += x
    print("weights: ", self.weights)

  def sync_update(self, test, cycle):
    c = 0
    while (c < cycle):
      copy_test = np.copy(test)
      for i in range(self.length):
        sigma = np.dot(self.weights[i], copy_test)
        # print("sigma", sigma)
        test[i] = 1 if (sigma * test[i] >= 0) else -1
      print("iteration:", c)
      print("after update: ", test)
      c += 1
      if np.array_equal(test, copy_test):
        print("test pattern is stable")
        return

      if c == (cycle - 3):
        t = test

    if np.array_equal(t, test):
      print("pattern is repeating with cycle of length 2")
###############################################################   
patterns = np.array([[1, 1, 1, -1, -1, -1],
                 [1, -1, 1, -1, 1, -1]])

test1 = np.array([1, 1, 1, -1, -1, -1])
test2 = np.array([-1, 1, 1, -1, -1, -1])

h = Hopfield(patterns)
cycle = 20
h.sync_update(test1, cycle)
print("************************************")
h.sync_update(test2, cycle)

