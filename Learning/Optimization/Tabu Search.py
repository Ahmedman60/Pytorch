
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# This code implementation based on book  Essentials of Metaheuristics by Sean Luke.
# The Code written and reviewed by Mohamed Fathallah.
# Define the Rastrigin function
# https://en.wikipedia.org/wiki/Rastrigin_function


def rastrigin(x, y):
    '''
    the Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
    It is a typical example of non-linear multimodal function.
    The Rastrigin function has many local minima, making it a good choice for demonstrating Tabu Search's effectiveness.
    '''
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

# Define the tweak function to generate a new candidate solution


def tweak(solution, step_size=0.5):
    # generate 2 random numbers and add them to the solution.
    return solution + np.random.uniform(-step_size, step_size, size=2)


# Tabu Search implementation
def tabu_search(max_iter=100, tabu_list_size=5, num_tweaks=10):
    """
    Performs Tabu Search optimization using the Rastrigin function.

    Parameters:
        max_iter (int): Maximum number of iterations.
        tabu_list_size (int): Maximum size of the Tabu list.
        num_tweaks (int): Number of candidate solutions to generate per iteration.

    Returns:
        best_solution (np.ndarray): The best solution found.
        solutions_history (list): History of solutions for visualization.
        best_history (list): History of the best solutions for visualization.
    """
    # Initialize variables
    current_solution = np.random.uniform(-5.12, 5.12, size=2)
    best_solution = current_solution.copy()

    tabu_list = []

    # Track progress for visualization
    solutions_history = []
    best_history = []

    for _ in range(max_iter):
        # Add the current solution to the Tabu list
        if len(tabu_list) >= tabu_list_size:
            tabu_list.pop(0)
        tabu_list.append(current_solution.tolist())

        # Generate candidates and evaluate them
        min_value = float('inf')
        best_candidate = current_solution
        for _ in range(num_tweaks):
            candidate = tweak(current_solution)
            if candidate.tolist() not in tabu_list:
                candidate_value = rastrigin(candidate[0], candidate[1])
                if candidate_value < min_value:
                    min_value = candidate_value
                    best_candidate = candidate

        # Update current solution
        if best_candidate.tolist() not in tabu_list:
            current_solution = best_candidate

        # Update best solution if applicable
        if rastrigin(current_solution[0], current_solution[1]) < rastrigin(best_solution[0], best_solution[1]):
            best_solution = current_solution

        # Record history for visualization
        solutions_history.append(current_solution)
        best_history.append(best_solution)

    return best_solution, solutions_history, best_history


# Run Tabu Search
best_solution, solutions_history, best_history = tabu_search()

# # Visualize the Rastrigin function and optimization process
# x = np.linspace(-5.12, 5.12, 400)
# y = np.linspace(-5.12, 5.12, 400)
# x, y = np.meshgrid(x, y)
# z = rastrigin(x, y)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

# # Plot the optimization path
# solutions_history = np.array(solutions_history)
# ax.plot(solutions_history[:, 0], solutions_history[:, 1], [rastrigin(
#     p[0], p[1]) for p in solutions_history], color='red', marker='o')

# # Highlight the best solution
# ax.scatter(best_solution[0], best_solution[1], rastrigin(
#     best_solution[0], best_solution[1]), color='blue', s=100, label='Best Solution')

# ax.set_title('Tabu Search Optimization on Rastrigin Function')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Rastrigin Value')
# plt.legend()
# plt.show()

# print("Best solution:", best_solution)
# print("Best value:", rastrigin(best_solution[0], best_solution[1]))


# plt.plot(best_history[0], rastrigin(
#     best_history[0], best_history[1]), '-o', color='blue')
# plt.xlabel('Iteration')
# plt.ylabel('Best Value')

# plt.show()


# # Plot Rastrigin values over iterations (x-axis)  --Not so informative
# plt.figure(figsize=(10, 6))
# x_values = [s[0] for s in solutions_history]
# rastrigin_values = [rastrigin(s[0], s[1]) for s in solutions_history]
# plt.plot(x_values, rastrigin_values, marker='o', color='purple')
# plt.title('Rastrigin Function Value vs X Over Iterations')
# plt.xlabel('X Value')
# plt.ylabel('Rastrigin Value')
# plt.grid()
# plt.show()

# print("Best solution:", best_solution)
# print("Best value:", rastrigin(best_solution[0], best_solution[1]))

# Plot Rastrigin values over iterations (iterations on x-axis)
plt.figure(figsize=(10, 6))
iterations = range(len(solutions_history))
rastrigin_values = [rastrigin(s[0], s[1]) for s in solutions_history]
plt.plot(iterations, rastrigin_values, marker='o', color='green')
plt.title('Rastrigin Function Value vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Rastrigin Value')
plt.grid()
plt.show()

print("Best solution:", best_solution)
print("Best value:", rastrigin(best_solution[0], best_solution[1]))
