# This algorithm is the exact same as np.random.choice(population, p=probabilities) directly samples from a given probability distribution.
'''
Algorithm 30: Fitness-Proportionate Selection (also known as Roulette Wheel Selection) is functionally similar to np.random.choice(list, p=probabilities) 
(which I provided before), 
but it follows a step-by-step approach to convert raw fitness values into a Cumulative Distribution Function (CDF) 
and select an individual based on a random number.
https://www.youtube.com/watch?v=kASLw8OWo8U&ab_channel=CikguAinan
'''


import numpy as np


def fitness_proportionate_selection(population, fitness):

    fitness = np.array(
        fitness, dtype=float)  # Ensure fitness is in float format

    # Handle all zero fitness case by setting uniform fitness
    if np.all(fitness == 0):
        fitness = np.ones_like(fitness)

    # Convert fitness to a CDF (cumulative sum)
    cdf = np.cumsum(fitness)
    total_fitness = cdf[-1]  # Last value is the total fitness sum

    # Select a random value in the range [0, total_fitness]
    n = np.random.uniform(0, total_fitness)

    # Find the first index where CDF surpasses n
    for i, value in enumerate(cdf):
        if n <= value:
            return population[i]

    return population[0]  # Fallback (should never happen)


# Example usage
population = ["A", "B", "C", "D"]
fitness = [10, 30, 50, 10]  # Higher fitness = higher selection chance

""" for example n is  27
10    0-10 A   this is 10%
40   11-40  B  this is 30%
90   41-90  C  this is 50%
100  90-100 D  this is 10% 
"""
print(np.cumsum(fitness)/sum(fitness))
selected = fitness_proportionate_selection(population, fitness)
print("Selected Individual:", selected)
# print(np.cumsum(fitness))
# # this this it should be  but nevertheless i don't want the probability
# print(np.cumsum(fitness)/sum(fitness))

# # if we make it using probability we will take random number from 0 and 1 .
# n = np.random.uniform(0, sum(fitness))

# print(n)
