import random

# Define the fitness function (to be customized based on the problem)


def fitness_function(individual):
    # Example: sum of genes as fitness - i need to modify this function to meet the requirements
    return sum(individual)

# Generate a random individual the individual are vectors.


def random_individual(size):
    return [random.randint(0, 1) for _ in range(size)]

# Selection function (Tournament selection)


def select_with_replacement(population, fitnesses):
    ''' random.choices
This function is part of Python's random module.

It performs a weighted random selection from the population based on the weights (fitnesses).

The weights parameter assigns a probability to each individual in the population. Individuals with higher fitness values have a higher chance of being selected.

The k=1 parameter specifies that only one individual should be selected.
[0]:Since random.choices returns a list (even when k=1), the [0] is used to extract the single selected individual (vector) from the list.
'''

    return random.choices(population, weights=fitnesses, k=1)[0]


# Testing select with replacement function

population = ['A', 'B', 'C', 'D']
# Higher fitness means higher probability of selection
fitnesses = [10, 20, 30, 40]

selected_individual = select_with_replacement(population, fitnesses)
selected_individual2 = random.choices(population, fitnesses, k=1)
# Output: One of 'A', 'B', 'C', or 'D', with 'D' being the most like
print(selected_individual)
print(selected_individual2)

# # Crossover function (Single-point crossover)


# def crossover(parent_a, parent_b):
#     point = random.randint(1, len(parent_a) - 1)
#     return parent_a[:point] + parent_b[point:], parent_b[:point] + parent_a[point:]

# # Mutation function (Bit flip mutation)


# def mutate(individual, mutation_rate=0.01):
#     return [gene if random.random() > mutation_rate else 1 - gene for gene in individual]

# # Genetic Algorithm


# def genetic_algorithm(pop_size, gene_size, max_generations):
#     population = [random_individual(gene_size) for _ in range(pop_size)]
#     best = None

#     for _ in range(max_generations):
#         fitnesses = [fitness_function(ind) for ind in population]
#         best_index = fitnesses.index(max(fitnesses))
#         best = population[best_index]

#         new_population = []
#         for _ in range(pop_size // 2):
#             parent_a = select_with_replacement(population, fitnesses)
#             parent_b = select_with_replacement(population, fitnesses)
#             child_a, child_b = crossover(parent_a, parent_b)
#             new_population.extend([mutate(child_a), mutate(child_b)])

#         population = new_population

#     return best


# # Run the Genetic Algorithm
# best_solution = genetic_algorithm(
#     pop_size=10, gene_size=8, max_generations=100)
# print("Best solution found:", best_solution)
