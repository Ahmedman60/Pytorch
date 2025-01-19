import torch
import seaborn as sns

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Get the current GPU device
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))


# YOU NEED TO INSTALL all the other modules for machine learning
