import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# (The rest of the code remains unchanged)


# Define the Rastrigin function


def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Evolution Strategy Algorithm with tracking


def evolution_strategy_visual(mu, lambd, dim, generations, bounds):
    # Initialize population
    population = [np.random.uniform(bounds[0], bounds[1], dim)
                  for _ in range(lambd)]
    best_solution = None
    best_fitness = float('inf')

    history = []  # To store population positions for visualization

    for gen in range(generations):
        # Assess fitness of each individual
        fitness = [rastrigin(ind) for ind in population]
        if min(fitness) < best_fitness:
            best_fitness = min(fitness)
            best_solution = population[np.argmin(fitness)]

        # Store current population for visualization
        history.append(np.array(population))

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

    return best_solution, best_fitness, history


# Parameters
mu = 5            # Number of parents
lambd = 20        # Number of children
# Dimensionality of the solution (must be 2 for 3D visualization)
dim = 2
generations = 50  # Number of generations
bounds = (-5.12, 5.12)  # Rastrigin bounds

# Run the algorithm
best_solution, best_fitness, history = evolution_strategy_visual(
    mu, lambd, dim, generations, bounds)

# # Create 3D Animation
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Generate Rastrigin surface
# x = np.linspace(bounds[0], bounds[1], 100)
# y = np.linspace(bounds[0], bounds[1], 100)
# x, y = np.meshgrid(x, y)
# z = 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + \
#     (y**2 - 10 * np.cos(2 * np.pi * y))

# ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
# points, = ax.plot([], [], [], 'ro', markersize=5, label="Population")

# # Set plot limits and labels
# ax.set_xlim(bounds[0], bounds[1])
# ax.set_ylim(bounds[0], bounds[1])
# ax.set_zlim(0, np.max(z))
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Fitness")
# ax.set_title("Evolution Strategy Optimization")
# ax.legend()

# # Update function for animation


# def update(frame):
#     current_population = history[frame]
#     x_vals = current_population[:, 0]
#     y_vals = current_population[:, 1]
#     z_vals = [rastrigin(ind) for ind in current_population]

#     points.set_data(x_vals, y_vals)
#     points.set_3d_properties(z_vals)
#     return points,


# # Create animation
# ani = FuncAnimation(fig, update, frames=len(history), interval=200, blit=True)

# # Save as GIF
# ani.save("evolution_strategy.gif", writer="pillow", fps=10)

# print("Animation saved as 'evolution_strategy.gif'.")
