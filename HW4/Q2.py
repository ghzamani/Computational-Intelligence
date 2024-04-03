#write your code here
import numpy as np
from numpy import random
import math
import random

class Equation:
  def __init__ (self, ch, population, generation):
    self.chromosome = ch
    self.population_size = population
    self.generation_size = generation

  def Value(self, x):
    return 9 * x**5 -194.7 * x**4 + 1680.1 * x**3 - 7227.94 * x**2 + 15501.2 * x - 13257.2

  def Generate(self):
    sol = random.choices(range(10), k=6)
    # sol[0] is even ? positive : negative
    return sol

  def ConvertListToNum(self, x):
    # i = int(''.join(map(str, x[1:2])))
    i = int(x[1])
    f = float(''.join(map(str, x[2:]))) * 0.001
    num = i + f
    if (x[0] % 2 == 1):
      num *= -1
    return num

  def Fitness(self, sol):
    x = self.ConvertListToNum(sol)
    return abs(self.Value(x))

  def Reproduction(self, fit, k):
    # use Elitist and choose the parents with better fitness (lower fitntess)
    idx = np.argpartition(fit, k)
    return idx[:k]

  def CrossOver(self, p1, p2):
    # use two-point crossover
    ch1 = np.copy(p1)
    ch2 = np.copy(p2)
    (i, j) = random.sample(range(self.chromosome), 2)
    
    if (i > j):
      for k in range(i, self.chromosome):
        (ch1[k], ch2[k]) = (ch2[k], ch1[k])

      for k in range(j+1):
        (ch1[k], ch2[k]) = (ch2[k], ch1[k])

    else:
      for k in range(i, j+1):
        (ch1[k], ch2[k]) = (ch2[k], ch1[k])

    return (ch1, ch2)

  def Mutation(self, m):
    (i, j) = random.sample(range(self.chromosome), 2)
    (m[i], m[j]) = (m[j], m[i])
    return m

  def GA(self):
    # initialize population 
    # and calculate fitness of each
    pop = []
    fitness = []

    for i in range(self.population_size):
      sol = self.Generate()
      pop.append(sol)
      fitness.append(self.Fitness(sol))

    # do the loop for "generation" times
    for g in range(self.generation_size):
      # do reproduction and choose parents
      k = 4
      indexes = self.Reproduction(fitness, k)
      parents = []
      for index in indexes:
        parents.append(pop[index])

      children = []
      # do crossover
      for i in range(len(parents)-1):
        for j in range(i+1, len(parents)):
          ch1, ch2 = self.CrossOver(parents[i], parents[j])
          children.append(ch1)
          children.append(ch2)
      
      pop = []
      fitness = []
      for p in parents:
        pop.append(p)
        fitness.append(self.Fitness(p))

      # do mutation for each new child
      for child in children:
        child = self.Mutation(child)
        pop.append(child)
        fitness.append(self.Fitness(child))

    best = np.argmin(fitness) 
    print("best solution is:", self.ConvertListToNum(pop[best]))
    print("f(solution) = ", fitness[best])

# 1 bit --> sign
# 1 + 4 bits --> float number
# -9.9999 to +9.9999
equ = Equation(6, 500, 2000)
equ.GA()

