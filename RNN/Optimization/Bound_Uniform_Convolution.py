import torch
import torch.nn as nn


# Vectorization Bound_Uniform_Convolution


def Bound_Uniform_Random(v, p, r, min, max):
    noise_Vector = torch.rand(1, len(v))
    mask = p > noise_Vector
    # we need to add randomness of sign  i don't use R here i use it inside.
    r = noise_Vector/2
    # we have the randomness
    noise = (r*(2*torch.randint(2, size=(1, len(v)))-1))
    vector = torch.where(mask, noise+v, v)
    vector = torch.clamp(vector, min=min, max=max)
    return vector


def Bound_Uniform_Random_For(v, p, r, min_val, max_val):
    # Initialize output vector
    vector = v.clone()

    for i in range(len(v)):
        # Step 7: if p ≥ random number chosen uniformly from 0.0 to 1.0
        if p >= torch.rand(1).item():
            while True:
                # Step 9: n ← random number chosen uniformly from -r to r inclusive
                n = (2 * torch.rand(1) - 1) * r
                # Step 10: check if min ≤ vi + n ≤ max
                if min_val <= vector[i] + n <= max_val:
                    # Step 11: vi ← vi + n
                    vector[i] = vector[i] + n
                    break

    return vector


# Initialize the values from the image
v = torch.tensor([5.1, 2.1, 4.2, 3.3, 7.0]).float()
p = 0.5  # probability threshold
r = 1.0/2  # Noise/2 as shown in image
min_val = 0.0
max_val = 10.0  # assuming this based on the values


def Bound_Uniform_Random(v, p, r, min_val, max_val):
    # Random numbers for probability check
    noise_Vector = torch.rand(1, len(v))
    mask = p > noise_Vector

    # Generate uniform noise from -r to r
    noise = (2 * torch.rand(1, len(v)) - 1) * r

    # Check if v + noise stays within bounds
    bounds_mask = (v + noise >= min_val) & (v + noise <= max_val)

    # Only apply noise where both probability condition and bounds conditions are met
    final_mask = mask & bounds_mask
    vector = torch.where(final_mask, v + noise, v)

    return vector


# Test the function
result = Bound_Uniform_Random(v, p, r, min_val, max_val)
print("Original vector:", v)
print("Result vector:", result)
