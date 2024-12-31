
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Define the Rastrigin function
# https://en.wikipedia.org/wiki/Rastrigin_function
def rastrigin(x, y):
    '''
    the Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
    It is a typical example of non-linear multimodal function.
    The Rastrigin function has many local minima, making it a good choice for demonstrating Tabu Search's effectiveness.
    '''
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
