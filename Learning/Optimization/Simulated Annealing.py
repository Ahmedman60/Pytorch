
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Objective function to minimize


def objective_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.1 * (x**2 + y**2)

# Simulated Annealing


def simulated_annealing(func, bounds, max_iter=500, temp=1.0, cooling_rate=0.99):
    x = np.random.uniform(bounds[0][0], bounds[0][1])
    y = np.random.uniform(bounds[1][0], bounds[1][1])
    current_solution = (x, y)
    current_value = func(x, y)

    best_solution = current_solution
    best_value = current_value

    history = [(x, y, current_value)]

    for i in range(max_iter):
        # Generate new candidate solution
        x_new = x + np.random.uniform(-0.5, 0.5)
        y_new = y + np.random.uniform(-0.5, 0.5)
        if not (bounds[0][0] <= x_new <= bounds[0][1] and bounds[1][0] <= y_new <= bounds[1][1]):
            continue  # Skip if out of bounds

        new_value = func(x_new, y_new)
        delta = new_value - current_value

        # Acceptance condition
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x, y = x_new, y_new
            current_value = new_value
            if current_value < best_value:
                # < because we want minimum value
                best_solution = (x, y)
                best_value = current_value

        history.append((x, y, current_value))
        temp *= cooling_rate  # Reduce temperature

    return best_solution, best_value, history


# Setting bounds and parameters
bounds = [(-10, 10), (-10, 10)]
best_sol, best_val, history = simulated_annealing(objective_function, bounds)


# This will create a 3D plot of the objective function and animate the optimization process. The red dot represents the current solution at each iteration.
# Create the 3D plot and animation
x_vals = np.linspace(bounds[0][0], bounds[0][1], 200)
y_vals = np.linspace(bounds[1][0], bounds[1][1], 200)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
z_vals = objective_function(x_mesh, y_mesh)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, y_mesh, z_vals, cmap='viridis', alpha=0.6)

# Initialize scatter plot
scat = ax.scatter([], [], [], color='red', s=50)


# # Update function for animation (It creates lots of frames, so it may take a while to render)
# def update(frame):
#     if frame < len(history):
#         # Update the scatter plot's position
#         x, y, z = history[frame]
#         # Update the scatter plot's position
#         scat._offsets3d = ([x], [y], [z])
#     return scat,


# # Create animation
# ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=False)

# # Save and display animation
# ani.save('simulated_annealing_optimization.gif', writer='pillow')
# plt.show()

# Ploting the loss function as well
# 3D Scatter Plot

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract parameters and loss values
x_history = [x for x, _, _ in history]
y_history = [y for _, y, _ in history]
loss_history = [z for _, _, z in history]

ax.scatter(x_history, y_history, loss_history,
           c='red', marker='o', label='Iterations')

# Adding labels and title
ax.set_title("Loss Function vs Parameters (Simulated Annealing)", fontsize=14)
ax.set_xlabel("Parameter X")
ax.set_ylabel("Parameter Y")
ax.set_zlabel("Loss (Objective Function Value)")

plt.legend()
plt.show()


# Loss information in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

plt.plot(x_history, y_history, loss_history,
         color='red', marker='o', label='Iterations')

# Adding labels and title
plt.title("Loss Function vs Parameters (Simulated Annealing)", fontsize=14)
plt.xlabel("Parameter X")

plt.ylabel("Parameter Y")

plt.legend()
plt.show()

# # Loss information in 2D with only one of the parameters x_history

plt.plot(loss_history,
         color='red', label='Iterations')

# Adding labels and title

plt.title("Loss Function vs Iterations (Simulated Annealing)", fontsize=14)
plt.xlabel("Iteration Number")
plt.ylabel("Loss (Objective Function Value)")

plt.legend()

plt.show()
