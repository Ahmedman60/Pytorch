import random
import numpy as np


def stochastic_universal_sampling(population, fitnesses, num_selected):
    # Convert fitnesses to CDF
    cumulative_fitness = np.cumsum(fitnesses)
    total_fitness = cumulative_fitness[-1]

    # Step size
    # you might pass 1 to select the top highest fitness if you use for loop in the algorithm
    # or you can select the pop_size//2 like selecting top 10 height of the population and step my step crossover them without loop.
    # for now we have this algorithm to select k individuals from the population based on highest fitness. and it is better and fair that roulette selection.
    step_size = total_fitness / num_selected

    # Random start point
    start_point = random.uniform(0, step_size)
    # for example we will have 3 pointers here if num_selected is 3
    pointers = [start_point + i * step_size for i in range(num_selected)]

# we select based on First CDF ≥ Pointer
    # Selection process
    selected = []
    index = 0
    for pointer in pointers:
        while cumulative_fitness[index] < pointer:
            index += 1
        selected.append(population[index])
        # you can remove the while loop and use below code  both easy to understand below code i mine.
        # for i, value in enumerate(cumulative_fitness):
        #     if value >= pointer:
        #         index = i
        #         break
    return selected


# Example Usage
population = ['A', 'B', 'C', 'D', 'E', 'F']
fitnesses = [10, 20, 30, 15, 5, 20]
num_selected = 3

selected_individuals = stochastic_universal_sampling(
    population, fitnesses, num_selected)
print("Selected Individuals:", selected_individuals)
