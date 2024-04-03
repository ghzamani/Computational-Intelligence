# Q1_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1_graded
# Do not change the above line.

# This cell is for your codes.

class StochasticGradientDescent:
  input_count = 0
  def __init__(self, dataSet, learning_rate, max_deltaW):
    self.dataSet = dataSet
    # rows of dataset count
    self.input_count = dataSet.shape[1]
    self.learning_rate = learning_rate
    self.max_deltaW = max_deltaW
    #define randomly initialized weights
    self.weights = np.random.rand(1, self.input_count+1)[0]

  #desired is nor
  def desired(self, x): return -1 if (1 in x[1:]) else 1

  def error(self, x, w): 
    d = self.desired(x)
    return d - np.dot(x, w)

  def iterations(self):
    deltaW = float("inf")
    count = 0

    #iterate on all datas while max change in weights is more than small number given
    while deltaW > self.max_deltaW :
      index = count % self.dataSet.shape[0]
      biased_data = np.insert(self.dataSet[index], 0, 1) 
      e = self.error(biased_data, self.weights)
      add_term = (self.learning_rate * e) * (biased_data)
      np.add(add_term, self.weights, self.weights)
      deltaW = np.max(np.abs(add_term))
      count += 1
      self.learning_rate *= 0.999

    return count


# Q1_graded
# Do not change the above line.

# This cell is for your codes.

data = np.array([[-1, -1],
                 [-1, 1], 
                 [1, -1],
                 [1, 1]])

gd = StochasticGradientDescent(data, 0.3, 0.0001) #(n, eta, small number)
i = gd.iterations()
print("Learning is done in ", i, " itetation.")
print("Weights are: ", gd.weights)


