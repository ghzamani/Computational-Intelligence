# Q2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.axis as ax

# Q2_graded
# Do not change the above line.

# This cell is for your codes.

class StochasticGradientDescent:
  input_count = 0
  def __init__(self, biased_data, desired, learning_rate, max_deltaW):
    self.biased_data = biased_data
    self.desired = desired
    self.input_count = biased_data.shape[1]
    self.learning_rate = learning_rate
    self.max_deltaW = max_deltaW
    #define randomly initialized weights
    self.weights = np.random.rand(1, self.input_count)[0]

  def error(self, x, w, d): return d - np.dot(x, w)

  def E(self, e): return 0.5 * (e**2)


  def iterations(self):
    err_per_iteration = []
    deltaW = float("inf")
    count = 0

    #iterate on all datas while max change in weights is more than small number given
    while deltaW > self.max_deltaW :
    # while count < 100:
      index = count % self.biased_data.shape[0]
      e = self.error(biased_data[index], self.weights, desired[index]) 
      tmp = np.copy(self.weights)
      add_term = (self.learning_rate * e) * (biased_data[index]) 
      np.add(add_term, self.weights, self.weights)
      min_weight = np.min(np.abs(self.weights))
      np.divide(self.weights, min_weight, self.weights)
      # deltaW = np.max(np.abs(add_term))
      deltaW = np.max(np.abs(np.subtract(self.weights, tmp)))
      #print("deltaW", deltaW)
      self.learning_rate *= 0.999

      err_per_iteration.append((count, self.E(e)))
      count += 1
      
    return err_per_iteration


# Q2_graded
# Do not change the above line.

# This cell is for your codes.
my_file = open("./Binary Classification/data.txt", "r")

#count of data
content = my_file.readlines()
lines_num = len(content)

dataSet = np.zeros((lines_num, 2))
biased_data = np.zeros((lines_num, 3))
desired = np.zeros((1, lines_num))[0]

#split each line 
i = 0
for line in content:
  l = line.split(",")
  l = [float(x) for x in l]
  # np.append(dataSet, [2,3])
  biased_data[i] = [1, l[0], l[1]]
  dataSet[i] = [l[0], l[1]]
  
  #change 0 to -1 in desired
  if l[2] == 0: desired[i] = -1
  else: desired[i] = 1
  i += 1

gd = StochasticGradientDescent(biased_data, desired, 0.3, 0.00001)
error_iter = gd.iterations()
print(gd.weights)


slope = -1 * gd.weights[1] / gd.weights[2]
point = (0, -1 * gd.weights[0] / gd.weights[2])

x1 = np.linspace(-200,-50,10000)
y1 = slope * x1 - gd.weights[0] / gd.weights[2]


colors = ['#1f77b4' if d == -1 else '#2ca02c' for d in desired]
x, y = dataSet.T
x2 = np.array([err[0] for err in error_iter])
y2 = np.log([err[1] for err in error_iter])
print(x2)
print(y2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.plot(x1, y1, '-r')
ax1.scatter(x, y, c=colors)
ax2.plot(x2, y2)
plt.figure(figsize=(16, 9))

