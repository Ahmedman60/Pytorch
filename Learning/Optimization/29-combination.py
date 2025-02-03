import numpy as np


def intermediate_recombination(v, w, p=0.25, bounds=(-10, 10)):
    """
    Perform Intermediate Recombination on two floating-point vectors v and w.

    Parameters:
        v (np.array): First parent vector.
        w (np.array): Second parent vector.
        p (float): Determines how far along the line a child can be located.
        bounds (tuple): The lower and upper bounds for valid values.

    Returns:
        tuple: Two child vectors after recombination.
    """
    l = len(v)
    new_v = np.copy(v)
    new_w = np.copy(w)

    for i in range(l):
        while True:
            # Random alpha from -p to 1 + p
            alpha = np.random.uniform(-p, 1 + p)
            # Random beta from -p to 1 + p
            beta = np.random.uniform(-p, 1 + p)

            t = alpha * v[i] + (1 - alpha) * w[i]
            s = beta * w[i] + (1 - beta) * v[i]

            # Ensure t and s are within bounds
            if bounds[0] <= t <= bounds[1] and bounds[0] <= s <= bounds[1]:
                new_v[i] = t
                new_w[i] = s
                break  # Move to the next element

    return new_v, new_w


# Example usage:
v_parent = np.array([3.5, -2.0, 4.8])
w_parent = np.array([-1.2, 5.1, -3.3])

child_v, child_w = intermediate_recombination(v_parent, w_parent, p=0.25)

print("Parent v:", v_parent)
print("Parent w:", w_parent)
print("Child v:", child_v)
print("Child w:", child_w)


'''
Normally, crossover only generates solutions within the range of the parents. But what if we need to explore beyond that range?

If p = 0, the new solutions stay within the bounding box of the parents.
If p > 0, we allow solutions outside the bounding box, helping the algorithm explore new areas.
Why is this useful?

If there is no mutation, the algorithm would be stuck within the bounds of the current population.
Allowing p > 0 helps the algorithm discover better solutions by looking beyond what it already knows.


'''
