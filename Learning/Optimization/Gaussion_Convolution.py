import numpy as np


def gaussian_convolution(vector, p=1.0, variance=1.0, min_val=-float('inf'), max_val=float('inf')):
    """
    Implement Gaussian Convolution on a vector

    Args:
        vector (list/array): Input vector to be convolved
        p (float): Probability of adding noise to an element (default: 1.0)
        variance (float): Variance of the Normal distribution (default: 1.0)
        min_val (float): Minimum allowed value for vector elements
        max_val (float): Maximum allowed value for vector elements

    Returns:
        numpy.array: Convolved vector
    """
    # Convert input to numpy array if it isn't already
    v = np.array(vector, dtype=float)

    # Process each element in the vector
    for i in range(len(v)):
        # Check if we should add noise to this element
        # this will always add noise to the element because p is 1.0.
        # if we decrease the probability of adding noise to the element, we can control the amount of noise added to the element.
        # if p is 0.5, then there is 50% chance of adding noise to the element.
        if np.random.random() < p:
            # Keep generating noise until we get a value within bounds
            while True:
                # Generate random noise from Normal(0, variance)
                # the norml takes means and std   std is np.sqrt(variance)
                n = np.random.normal(0, np.sqrt(variance))
                # Check if adding the noise keeps us within bounds
                if min_val <= v[i] + n <= max_val:
                    v[i] += n
                    break

    return v


# Example usage
np.random.seed(42)  # For reproducibility

# Create a sample vector
original_vector = [1.0, 2.0, 3.0, 4.0, 5.0]

# Example 1: Basic convolution with standard normal distribution
result1 = gaussian_convolution(
    vector=original_vector,
    p=1.0,
    variance=1.0
)

# Example 2: Convolution with bounds
result2 = gaussian_convolution(
    vector=original_vector,
    p=1.0,
    variance=2.0,
    min_val=0.0,
    max_val=6.0
)

# Print results
print("Original vector:", original_vector)
print("\nExample 1 - No bounds:")
print("Convolved vector:", result1.round(3))
print("Change in values:", (result1 - original_vector).round(3))

print("\nExample 2 - With bounds [0, 6]:")
print("Convolved vector:", result2.round(3))
print("Change in values:", (result2 - original_vector).round(3))
