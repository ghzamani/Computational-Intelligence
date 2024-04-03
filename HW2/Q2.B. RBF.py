#write your code here
import numpy as np
import matplotlib.pyplot as plt

class RBF:
  def __init__ (self, k, learning_rate, epochs):
    self.kernel = k
    self.learning_rate = learning_rate
    self.epochs = epochs
    # self.usestd = usestd

    # random from “standard normal” distribution
    self.weights = np.random.randn(k)
    self.b = np.random.randn(1)
    self.centers = []
    self.radius = []
#####################################################################

  def train(self, data, desired):
    #find centers and radius by k-mean clustering
    self.centers, self.radius = self.k_cluster(data)
    for i in range(self.epochs):
      for j in range(data.shape[0]):
        #join centers and radius
        radials = np.array((self.centers, self.radius)).T
        #forward pass
        rbf = np.array([self.gaussian_rbf(data[j], center, r) for center, r in radials])
        sigma = rbf.T.dot(self.weights) + self.b

        #back propagate
        e = (desired[j] - sigma)
        Error = e **2
        print("train error ", Error[0])

        #update
        self.weights += self.learning_rate * rbf * e
        self.b += self.learning_rate * e

#####################################################################

  def gaussian_rbf(self, x, t, sigma):
    return np.exp(-1 * (x-t)**2 / (sigma**2))
#####################################################################

  def k_cluster(self, x):
    #randomly initialize centroids
    centroids = np.random.choice(np.squeeze(x), size=self.kernel, replace=False)

    convergence = False
    x = x.reshape((x.shape[0], 1))
    repeated_arr = np.repeat(x, self.kernel, axis=1)

    while not convergence:
      c = np.tile(centroids, (x.shape[0], 1))
      dist = np.abs(np.subtract(repeated_arr, c))
      index = np.argmin(dist, axis=1) #showing each point belongs to which cluster
      
      new_centroids = np.copy(centroids)
      for i in range(self.kernel):
        clusterpoints = x[index==i]
        if len(clusterpoints) != 0:
          new_centroids[i] = np.average(clusterpoints)
          
      #check for convergence
      max_change = np.amax(np.abs(np.subtract(new_centroids, centroids)))
      if max_change < 1e-6:
        convergence = True
      centroids = new_centroids

    #check the clusters
    c = np.tile(centroids, (x.shape[0], 1))
    dist = np.abs(np.subtract(repeated_arr, c))
    index = np.argmin(dist, axis=1)

    smallclusters = []
    standard_deviations = np.zeros(self.kernel)
    for i in range(self.kernel):
      clusterpoints = x[index==i]
      if len(clusterpoints) < 2:
        smallclusters.append(i) #clusters that have less than 2 points
        continue
      standard_deviations[i] = np.std(clusterpoints)

    # for clusters with less than 2 points, take the mean std of the other clusters
    if len(smallclusters) != 0:
      otherpoints = []
      for i in range(self.kernel):
        if i in smallclusters: continue
        otherpoints.append(x[index==i])
        otherpoints = np.reshape(1, -1)
      
      standard_deviations[smallclusters] = np.average(np.std(otherpoints))
    print("centroids: ", centroids)
    print("radiuses: ", standard_deviations)
    return (centroids, standard_deviations)

#####################################################################
  
  def predict(self, testdata):
    predictions = []
    for t in testdata:
      radials = np.array((self.centers, self.radius)).T
      rbf = np.array([self.gaussian_rbf(t, center, r) for center, r in radials])
      sigma = rbf.T.dot(self.weights) + self.b
      predictions.append(sigma)
    return np.array(predictions)

#train data
x_train = np.arange(-300, 300) / 50
noise = np.random.uniform(-0.3, 0.3, x_train.size)
y = np.sin(x_train) + noise
rbf = RBF(4, 0.01, 300)
rbf.train(x_train, y)

#test data
x_test = np.arange(-150, 150) / 50
predictions = rbf.predict(x_test)

plt.plot(x_test, predictions)
plt.plot(x_test, np.sin(x_test))
plt.legend(['prediction', 'sin'])
plt.show()


