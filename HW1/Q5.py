# Q5_graded
# Do not change the above line.

# This cell is for your imports

import numpy as np
import matplotlib.pyplot as plt


# Q5_graded
# Do not change the above line.

# This cell is for your codes.s

#implement the class with one hidden layer    
class MLP:
  def __init__(self, input_neurons, hidden_neurons, output_neurons, inputs, learning_rate):
    #neurons count
    self.input_neurons = input_neurons
    self.hidden_neurons = hidden_neurons
    self.output_neurons = output_neurons
    self.learning_rate = learning_rate

    #input data
    self.inputs = inputs

    #initialize weights between input layer and hidden layer
    self.w = np.random.rand(self.hidden_neurons, self.input_neurons)
    self.b1 = np.random.rand(self.hidden_neurons, 1)   

    #initialize weights between hidden layer and output layer
    self.wPrime = np.random.rand(self.output_neurons, self.hidden_neurons)
    self.b2 = np.random.rand(self.output_neurons, 1)

    #errors
    self.error = []

#####################################################################################

  def E(self, y, desired): return np.multiply(0.5, np.sum(np.power(np.subtract(desired, y), 2)))
    
  #activation functions
  def sigmoid(self, x): return 1 / (1 + np.exp(np.multiply(-1, x)))

  def softmax(self, net):
      a = np.exp(net)
      b = np.sum(a)
      return np.divide(a, b)

#####################################################################################
  def forwardpass(self, data):
    net1 = np.dot(self.w, data) + self.b1
    #use sigmoid for hidden layer
    out1 = self.sigmoid(net1)    
    net2 = np.dot(self.wPrime, out1) + self.b2
    #use softmax for output layer
    out2 = self.softmax(net2)    
    return (net1, out1, net2, out2)

#####################################################################################
  #backward pass
  def deltachange(self, data, target, out2, out1):
    #weight changes for output node
    term1 = np.subtract(out2, target)
    
    #derivative of sigmoid function
    # term2 = np.multiply(out2, np.subtract(1, out2))
    #derivative of softmax function
    term2 = np.multiply(out2, np.subtract(1, out2))    
    
    term3 = out1.T
    # print("term3", term3)
    x = np.multiply(term1, term2)
    # print("x", x)
    derivative = np.dot(x, term3) #rond(E_total) / rond(w_i)
    # print("derivative", derivative)
    new_wprime =  np.subtract(self.wPrime, np.multiply(self.learning_rate, derivative))
    # print("new weight primes", new_wprime)

    #update bias weight
    #just term3 changes
    #w_i * h1 + w_j * h2 + b ---> derivative = 1
    np.subtract(self.b2, np.multiply(self.learning_rate, x), self.b2)

    ##########################################
    #weight changes for hidden node
    # print("weight", self.w)
    term11 = np.dot(self.wPrime.T, x)
    #derivative of sigmoid function
    term22 = np.multiply(out1, np.subtract(1, out1))
    term33 = data.T
    x2 = np.multiply(term11, term22)
    derivative2 = np.dot(x2, term33) #rond(E_total) / rond(w_i)
    new_w = np.subtract(self.w, np.multiply(self.learning_rate, derivative2))
    # print("new weights", new_w)

    #update bias weight
    #just term3 changes
    #w_i * 1 + w_j * i2 + b ---> derivative = 1
    np.subtract(self.b1, np.multiply(self.learning_rate, x2), self.b1)
    # print("new bias", self.b1)

    #now update weights
    self.wPrime = new_wprime
    self.w = new_w

#####################################################################################
  #calculate forward and backward on each data
  def forwardbackward(self, line):
    l = line.split(",")
    l = [int(x) for x in l]
    desired = np.zeros((10, 1))
    desired[l[0]] = 1
    #normalize data
    data = np.divide(np.array(l[1:]).reshape(784, 1), 255)
    #forward
    (net1, out1, net2, out2) = self.forwardpass(data)
    
    
    self.error.append(self.E(out2, desired))
    #backward
    self.deltachange(data, desired, out2, out1)

#####################################################################################
  def train(self):  
    for line in self.inputs:
      self.forwardbackward(line)


  def test(self, testdata):
    true = 0
    for line in testdata:
      l = line.split(",")
      l = [int(x) for x in l]
      desired = l[0]
      #normalize data
      data = np.divide(np.array(l[1:]).reshape(784, 1), 255)
      #forward
      (net1, out1, net2, out2) = self.forwardpass(data)
      
      for j in range(len(out2)):
        if out2[j] == 1:
          break
      if j == desired:
        true += 1
    return true   




# Q5_graded
# Do not change the above line.

# This cell is for your codes.


my_file = open("./sample_data/mnist_train_small.csv", "r")

#count of data
content = my_file.readlines()
mlp = MLP(784, 128, 10, content, 0.3)

my_file = open("./sample_data/mnist_test.csv", "r")
#count of data
content2 = my_file.readlines()

train_errors = []
test_acc = []
#epoch = 3
mlp.train()
mse = np.sum(mlp.error) / 20000
train_errors.append(mse)
print("loss of training epoch1:", mse)

accuracy = mlp.test(content2)
test_acc.append(accuracy)
print("accuracy of test data", (accuracy / 10000))

mlp.train()
mse = np.sum(mlp.error) / 40000
train_errors.append(mse)
print("loss of training epoch2:", mse)

accuracy = mlp.test(content2)
test_acc.append(accuracy)
print("accuracy of test data", (accuracy / 10000))

mlp.train()
mse = np.sum(mlp.error) / 60000
train_errors.append(mse)
print("loss of training epoch3:", mse)

accuracy = mlp.test(content2)
test_acc.append(accuracy)
print("accuracy of test data", (accuracy / 10000))

# accuracy = mlp.test(content2)
# print("accuracy of test data", accuracy / 10000)

epoch = range(1, 4)
plt.plot(epoch, train_errors, label="train Loss")
plt.plot(epoch, test_acc, '-r')
plt.show()


