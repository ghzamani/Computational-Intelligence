# Q1.3_graded
# Do not change the above line.

# This cell is for your imports.
import numpy as np
from PIL import Image, ImageFont
import random
from random import randrange
import matplotlib.pyplot as plt


# download the font and move it to its folder
!wget "https://www.freebestfonts.com/yone//down/arial.ttf"
!mv arial.ttf /usr/share/fonts/truetype

# Q1.3_graded
# Do not change the above line.

# This cell is for your codes.
class Hopfield:
  def __init__(self, patterns, cycle):
    self.patterns_count = len(patterns) #number of patterns
    self.length = patterns[0].size  #length of each pattern
    self.weights = np.zeros((self.length, self.length)) #2d matrix of weights
    self.patterns = patterns
    self.cycle = cycle
    self.calculate_weights()

  def calculate_weights(self):
    for pattern in self.patterns:
      transpose = pattern.reshape(-1, 1)
      self.weights += pattern * transpose
    np.fill_diagonal(self.weights, 0)
    # print("weights: ", self.weights)

  def E (self, x):
    return -0.5 * np.dot(np.dot(self.weights, x), x.T)

  def activation(self, w, x, theta):
    sigma = np.dot(w, x) - theta
    return 1 if (sigma >= 0) else -1

  def async_update(self, test):
    output = np.copy(test)
    energy = []
    for i in range(self.cycle):
      rnd = random.randint(0, self.length-1)
      activation = self.activation(self.weights[rnd], output, 0)
      output[rnd] = activation
      energy.append(self.E(output))
    return (output, energy)

##############################################################################

#noisy images of data
def noisy_data(image, prob):
  output = np.zeros(image.shape, np.uint8)
  thresh = 1 - prob
  for i in range(image.shape[0]):
      for j in range(image.shape[1]):
          rdn = random.random()
          # noise = randrange(255)
          flip = random.randint(0, 1)
          n = [0, 255]
          if rdn < prob:
              output[i][j] = n[flip]
          elif rdn > thresh:
              output[i][j] = n[flip]
          else:
              output[i][j] = image[i][j]
  return output

def save_images(font_size):
  images = []
  font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", font_size)
  for char in "ABCDEFGHIJ":
    im = Image.Image()._new(font.getmask(char))
    im.save(f"{char}_{font_size}.bmp")
    images.append(Image.open(f"{char}_{font_size}.bmp"))
  return images

def binary_vector(v):
  # fill black pixels with 1
  output = np.full(network_size, 1)
  # fill white pixels with -1
  output[v == 0] = -1
  return output.flatten()

def data_test(images, network_size):
  patterns = []
  tests10 = []
  tests30 = []
  tests60 = []

  for im in images:
    # make array from data
    matrix = np.array(im.resize((network_size[1], network_size[0])))
    pattern = binary_vector(matrix)
    patterns.append(pattern)

    # make testdata too
    t = noisy_data(matrix, 0.1)
    test = binary_vector(t)
    tests10.append(test)

    t = noisy_data(matrix, 0.3)
    test = binary_vector(t)
    tests30.append(test)

    t = noisy_data(matrix, 0.6)
    test = binary_vector(t)
    tests60.append(test)

  return (patterns, tests10, tests30, tests60)

def convert_pattern(p):
  img = np.full(p.shape, 255)
  img[p == -1] = 0
  return img

def print_output(test, ch, percent, output, energy, network_size):
  i = convert_pattern(test)
  title = str(percent) + "% noisy, letter " + ch
  i = np.reshape(i, network_size)
  plt.imshow(i)
  plt.title(title)
  plt.show()

  output = np.reshape(output, network_size)
  plt.imshow(output)
  plt.title('hopfield output')
  plt.show()
  cycle = range(0, cycles)
  plt.plot(cycle, energy, label="energy")
  plt.show()

def accuracy(output, pattern):
  a = np.sum(output == pattern)
  b = np.sum(output != pattern)
  c = a if (a > b) else b
  return c * 100 / output.shape

##############################################################################

fonts = [16, 32, 64]
for font in fonts:
  print("font size:", font)
  # save image with its font
  images = save_images(font)
  # find size of the hopfield
  network_size = max([x.size for x in images])
  print("network size is", network_size)
  # make learn-data and test-data(with different noises) as array
  (patterns, tests10, tests30, tests60) = data_test(images, network_size)
  # train the network with learn-data
  cycles = 1500
  h = Hopfield(patterns, cycles)
  # test network with different percentages of noise
  chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  accuracies = []
  # noise 10%
  for idx in range(len(tests10)):
    (output, energy) = h.async_update(tests10[idx])
    print_output(tests10[idx], chars[idx], 10, output, energy, network_size)
    acc = accuracy(output, patterns[idx])
    accuracies.append(acc)
    print("letter", chars[idx], "accuracy is:", acc)
  
  print("font size", font, "total accuracy:", np.average(accuracies))
  print("**********************************************************")

  # noise 30%
  accuracies = []
  for idx in range(len(tests30)):
    (output, energy) = h.async_update(tests30[idx])
    print_output(tests30[idx], chars[idx], 30, output, energy, network_size)
    acc = accuracy(output, patterns[idx])
    accuracies.append(acc)
    print("letter", chars[idx], "accuracy is:", acc)
  
  print("font size", font, "total accuracy:", np.average(accuracies))
  print("**********************************************************")
  
  # noise 60%
  accuracies = []
  for idx in range(len(tests60)):
    (output, energy) = h.async_update(tests60[idx])
    print_output(tests60[idx], chars[idx], 60, output, energy, network_size)
    acc = accuracy(output, patterns[idx])
    accuracies.append(acc)
    print("letter", chars[idx], "accuracy is:", acc)
  
  print("font size", font, "total accuracy:", np.average(accuracies))
  print("**********************************************************")
  # break
  


