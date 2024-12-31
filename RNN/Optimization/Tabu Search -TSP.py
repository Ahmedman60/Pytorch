import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the distance function for TSP


def calculate_distance(path, distance_matrix):
    return np.sum([distance_matrix[path[i - 1], path[i]] for i in range(len(path))])

# Define the tweak function to generate a new candidate solution for TSP


def tweak_tsp(solution):
    new_solution = solution.copy()
    # Swap two random cities in the solution
    i, j = np.random.choice(len(solution), 2, replace=False)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

# Tabu Search implementation for TSP


def tabu_search_tsp(distance_matrix, max_iter=100, tabu_list_size=5, num_tweaks=10):
    # Initialize variables
    num_nodes = distance_matrix.shape[0]
    current_solution = list(np.random.permutation(num_nodes))
    best_solution = current_solution.copy()
    tabu_list = []

    # Track progress for visualization
    solutions_history = []
    best_history = []

    for _ in range(max_iter):
        # Add the current solution to the Tabu list
        if len(tabu_list) >= tabu_list_size:
            tabu_list.pop(0)
        tabu_list.append(current_solution)

        # Generate candidates and evaluate them
        candidates = [tweak_tsp(current_solution) for _ in range(num_tweaks)]
        candidates = [c for c in candidates if c not in tabu_list]

        if not candidates:
            break

        # Select the best candidate
        best_candidate = min(
            candidates, key=lambda c: calculate_distance(c, distance_matrix))

        # Update current solution
        current_solution = best_candidate

        # Update best solution if applicable
        if calculate_distance(current_solution, distance_matrix) < calculate_distance(best_solution, distance_matrix):
            best_solution = current_solution

        # Record history for visualization
        solutions_history.append(current_solution)
        best_history.append(best_solution)

    return best_solution, solutions_history, best_history


# Example TSP problem with 5 nodes
distance_matrix = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
])
# 10 5 8 4 2 this is 28

# Run Tabu Search for TSP
best_solution, solutions_history, best_history = tabu_search_tsp(
    distance_matrix)

# Visualize the TSP solution
node_coordinates = np.array([
    # Example coordinates for visualization
    [0, 0], [2, 2], [9, 0], [10, 10], [7, 5]
])

fig, ax = plt.subplots(figsize=(10, 6))

# Initialize the plot


def init():
    ax.clear()
    ax.set_title('TSP Solution Using Tabu Search')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    for i, coord in enumerate(node_coordinates):
        ax.text(coord[0], coord[1], f'Node {i}', fontsize=12, color='red')
    ax.grid()

# Update the plot for each frame


def update(frame):
    ax.clear()
    path = np.array(
        node_coordinates[np.array(solutions_history[frame] + [solutions_history[frame][0]])])
    ax.plot(path[:, 0], path[:, 1], alpha=0.7, linestyle='-',
            color='red',
            label=f'Current Path (Distance: {calculate_distance(solutions_history[frame], distance_matrix)})')
    best_path = np.array(
        node_coordinates[best_history[frame] + [best_history[frame][0]]]
    )
    ax.plot(
        best_path[:, 0], best_path[:, 1],
        marker='o', color='blue',
        label=f'Best Path (Distance: {calculate_distance(best_history[frame], distance_matrix)})'
    )
    for i, coord in enumerate(node_coordinates):
        ax.text(coord[0], coord[1], f'Node {i}', fontsize=12, color='red')
    ax.legend(loc='upper right')  # Move the legend to the upper right corner
    ax.grid()


ani = FuncAnimation(fig, update, frames=len(
    solutions_history), init_func=init, repeat=False)

# Save as GIF
ani.save('tsp_tabu_search.gif', writer='imagemagick', fps=2)

# Check if ImageMagick is installed
if shutil.which('convert'):
    ani.save('tsp_tabu_search.gif', writer='imagemagick', fps=2)
else:
    ani.save('tsp_tabu_search.gif', writer='pillow', fps=2)
plt.show()


print("Best solution:", best_solution)
print("Best path distance:", calculate_distance(best_solution, distance_matrix))


# print(sum([distance_matrix[3, 2], distance_matrix[2, 4],
#           distance_matrix[4, 0], distance_matrix[0, 1], distance_matrix[1, 3]]))
