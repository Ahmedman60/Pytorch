import numpy as np
import matplotlib.pyplot as plt

# Define the distance function for TSP


def calculate_distance(path, distance_matrix):
    return sum(distance_matrix[path[i - 1], path[i]] for i in range(len(path)))

# Define the tweak function to generate a new candidate solution for TSP


def tweak_tsp(solution):
    new_solution = solution.copy()
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

# Run Tabu Search for TSP
best_solution, solutions_history, best_history = tabu_search_tsp(
    distance_matrix)

# Visualize the TSP solution
node_coordinates = np.array([
    # Example coordinates for visualization
    [0, 0], [2, 2], [9, 0], [10, 10], [7, 5]
])

plt.figure(figsize=(10, 6))
for solution in solutions_history:
    path = np.array(node_coordinates[solution + [solution[0]]])
    plt.plot(path[:, 0], path[:, 1], alpha=0.3, linestyle='--', color='gray')

# Plot best solution
best_path = np.array(node_coordinates[best_solution + [best_solution[0]]])
plt.plot(best_path[:, 0], best_path[:, 1],
         marker='o', color='blue', label='Best Path')

for i, coord in enumerate(node_coordinates):
    plt.text(coord[0], coord[1], f'Node {i}', fontsize=12, color='red')

plt.title('TSP Solution Using Tabu Search')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid()
plt.show()

print("Best solution:", best_solution)
print("Best path distance:", calculate_distance(best_solution, distance_matrix))
