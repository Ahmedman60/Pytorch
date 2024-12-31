
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
