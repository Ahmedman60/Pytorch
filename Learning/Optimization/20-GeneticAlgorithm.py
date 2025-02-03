import random

# Define the fitness function (to be customized based on the problem)


def fitness_function(individual):
    # Example: sum of genes as fitness - i need to modify this function to meet the requirements
    return sum(individual)

# Generate a random individual the individual are vectors.


def random_individual(size):
    # The vector will be 0 and 1's
    return [random.randint(0, 1) for _ in range(size)]

# Selection function (Tournament selection)


def select_with_replacement(population, fitnesses):
    '''
Select with replacement means we can select the same individual again.
Since selection is "with replacement," the same individual can be selected multiple times.
random.choices
This function is part of Python's random module.
It performs a weighted random selection from the population based on the weights (fitnesses).
The weights parameter assigns a probability to each individual in the population. Individuals with higher fitness values have a higher chance of being selected.
The k=1 parameter specifies that only one individual should be selected.
[0]:Since random.choices returns a list (even when k=1), the [0] is used to extract the single selected individual (vector) from the list.
'''

    return random.choices(population, weights=fitnesses, k=1)[0]


# # Crossover function (Single-point crossover)


def crossover(parent_a, parent_b):
    # this is one point crossover  start from index 2 at least swap 2 like algorithm 23
    point = random.randint(1, len(parent_a) - 1)  # random crossover point
    return parent_a[:point] + parent_b[point:], parent_b[:point] + parent_a[point:]


def crossover_twopoints(parent_a, parent_b):
    point1 = random.randint(1, len(parent_a) - 2)
    point2 = random.randint(point1 + 2, len(parent_a) - 1)

    child1 = parent_a[:point1] + parent_b[point1:point2]+parent_a[point2:]
    child2 = parent_b[:point1] + parent_a[point1:point2]+parent_b[point2:]

    return child1, child2


def crossover_uniform(parent_a, parent_b):
    '''
    In theory, you could perform uniform crossover with several vectors at once to produce children
 whicharethecombinationofallofthem
    '''
    # Uniform crossover between two parents.
    child1 = []
    child2 = []
    for i in range(len(parent_a)):
        if random.random() < 0.5:
            child1.append(parent_a[i])
            child2.append(parent_b[i])
        else:
            child1.append(parent_b[i])
            child2.append(parent_a[i])

    return child1, child2

# # Mutation function (Bit flip mutation)


def mutate(individual, mutation_rate=0.01):
    ''''
    iterate over all individual and choice the same gene by probability of 0.99 and flip it by probability of 0.1.
    the flip  is 1-gene
    if  0   1-0  =1
    if  1   1-1  =0
    this is the same as the we did in select with replacement.
    '''

    # [gen if random.random() > mutation_rate else 1-gen for gen in individual]
    return [random.choices([gene, 1 - gene], weights=[1 - mutation_rate, mutation_rate])[0] for gene in individual]

# # Genetic Algorithm


def genetic_algorithm(pop_size, gene_size, max_generations):
    population = [random_individual(gene_size) for _ in range(pop_size)]
    best = None

    for _ in range(max_generations):
        fitnesses = [fitness_function(ind) for ind in population]
        best_index = fitnesses.index(max(fitnesses))
        best = population[best_index]

        new_population = []
        for _ in range(pop_size // 2):
            # pop_size//2  this because each iteration will create 2 children so i need to keep population fixed. by pop//2
            # This algorithm can make cross-over between same parents. or can use parent already mated with other.
            # it doesn't remove the parent from the population after cross-over which can cause it to mate again.
            parent_a = select_with_replacement(population, fitnesses)
            parent_b = select_with_replacement(population, fitnesses)
            child_a, child_b = crossover_uniform(parent_a, parent_b)
            new_population.extend([mutate(child_a), mutate(child_b)])

        population = new_population

    return best, len(population)


# Run the Genetic Algorithm
best_solution = genetic_algorithm(
    pop_size=10, gene_size=8, max_generations=100)
print("Best solution found:", best_solution)


'''
The current implementation replaces the entire population with only the newly generated children,
which can reduce genetic diversity over generations.
A better approach is to incorporate elitism or ensure that part of the new population includes top individuals
from the previous generation.


Expected Behavior:
1-Start with a population of popsize individuals.
2-Selection: Select two parents from the current population.
3-Crossover: Generate two children from the selected parents.
4-Mutation: Apply mutation to each child.
Repeat this process popsize/2 times to generate exactly popsize offspring.

Replace the old population with the new one. (generational algorithm) which update the entire sample once per iteration
---steady-state
update the sample a few candidate solutions at a time

Possible Issue:
The current implementation iterates correctly but may not be preserving diversity well.
It does not mix parents and offspring (pure generational replacement instead of steady-state GA).
'''
