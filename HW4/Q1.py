#write your code here
import numpy as np
from numpy import random
import math
import random

class TSP:
  def __init__ (self, cities_count, dists, population, generation):
    self.distances = dists
    self.count = cities_count
    self.population_size = population
    self.generation_size = generation

  def Generate(self):
    arr = np.arange(0, self.count)
    random.shuffle(arr)
    sol = np.append(arr, arr[0])
    return sol

  def Fitness(self, sol):
    total = [self.distances[sol[i], sol[i+1]] for i in range(self.count)]
    # -1 means there's no way, so the solution is not acceptable
    if -1 in total:
      return math.inf
    return sum(total)

  def Reproduction(self, fit, k):
    # use Elitist and choose the parents with better fitness (lower fitntess)
    idx = np.argpartition(fit, k)
    # return a list of parents' index
    return idx[:k]

  def CrossOver(self, p1, p2):
    # use order1 crossover
    child = []
    (i, j) = random.sample(range(self.count), 2)
    a = min(i, j)
    b = max(i, j)

    for i in range(a, b):
      child.append(p1[i])
    c = [item for item in p2 if item not in child]
    child.extend(c)
    
    if (len(child) != self.count + 1):
      child.append(child[0])
    else: child[len(child) -1 ] = child[0]
    
    return child

  def Mutation(self, m):
    (i, j) = random.sample(range(self.count), 2)
    
    (m[i], m[j]) = (m[j], m[i])
    m[len(m) -1] = m[0]
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
      k = 4 # number of parents
      indexes = self.Reproduction(fitness, k)
      parents = []
      for index in indexes:
        parents.append(pop[index])

      children = []
      # do crossover
      for i in range(len(parents)):
        for j in range(len(parents)):
          if j == i:
            continue
          child = self.CrossOver(parents[i], parents[j])
          children.append(child)
      
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

      if g % 40 == 0:
        best = np.argmin(fitness) 
        answer = [x+1 for x in pop[best]]
        print("best solution, iteration", g, ":", answer)
        print("length", fitness[best])

    best = np.argmin(fitness) 
    answer = [x+1 for x in pop[best]]
    print("best solution is:", answer)
    print("length:", fitness[best])

dists = np.array([[0, 12, 10, -1, -1, -1, 12],
                  [12, 0, 8, 12, -1, -1, -1],
                  [10, 8, 0, 11, 3, -1, 9],
                  [-1, 12, 11, 0, 11, 10, -1],
                  [-1, -1, 3, 11, 0, 6, 7],
                  [-1, -1, -1, 10, 6, 0, 9],
                  [12, -1, 9, -1, 7, 9, 0]
                  ])

tsp = TSP(7, dists, 10, 200)
tsp.GA()


