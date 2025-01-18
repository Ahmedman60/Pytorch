import numpy as np

# Define the Rastrigin function as the objective function


def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Evolution Strategy Algorithm


def evolution_strategy(mu, lambd, dim, generations, bounds):
    # Initialize population
    population = [np.random.uniform(bounds[0], bounds[1], dim)
                  for _ in range(lambd)]
    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        # Assess fitness of each individual
        fitness = [rastrigin(ind) for ind in population]
        if min(fitness) < best_fitness:
            best_fitness = min(fitness)
            best_solution = population[np.argmin(fitness)]

        # Select the top μ individuals
        selected_indices = np.argsort(fitness)[:mu]
        selected_parents = [population[i] for i in selected_indices]

        # Generate offspring (λ individuals) via mutation
        offspring = []
        for parent in selected_parents:
            for _ in range(lambd // mu):
                # Mutate by adding Gaussian noise
                child = parent + np.random.normal(0, 1, dim)
                child = np.clip(child, bounds[0], bounds[1])  # Ensure bounds
                offspring.append(child)

        # Replace population with offspring
        population = offspring

        # Print progress
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness


# Parameters
mu = 5            # Number of parents
lambd = 20        # Number of children
dim = 5           # Dimensionality of the solution
generations = 50  # Number of generations
bounds = (-5.12, 5.12)  # Rastrigin bounds

# Run the algorithm
best_solution, best_fitness = evolution_strategy(
    mu, lambd, dim, generations, bounds)

print("\nBest Solution:", best_solution)
print("Best Fitness:", best_fitness)
