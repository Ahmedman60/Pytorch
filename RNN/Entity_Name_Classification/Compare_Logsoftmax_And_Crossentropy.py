import torch
import torch.nn as nn

# # Example model
# model = nn.Sequential(
#     # Linear layer with 10 input features and 5 output classes
#     nn.Linear(10, 5),
#     nn.LogSoftmax(dim=1)  # Apply log_softmax
# )

# # Loss function
# criterion = nn.NLLLoss()

# # Example data
# inputs = torch.randn(3, 10)  # Batch of 3 samples, 10 features each
# targets = torch.tensor([1, 0, 4])  # Target class indices

# # Forward pass
# outputs = model(inputs)  # Outputs log-probabilities
# loss = criterion(outputs, targets)  # Compute NLL loss

# print("Loss:", loss.item())


# Alternative with CrossEntropyLoss
# If you prefer, you can skip log_softmax in the model and directly use CrossEntropyLoss, which computes both the softmax and the log-loss internally:
model = nn.Sequential(
    nn.Linear(10, 5),
)

criterion = nn.CrossEntropyLoss()

inputs = torch.randn(3, 10)  # Batch of 3 samples, 10 features each
targets = torch.tensor([1, 0, 4])  # Target class indices
loss = criterion(inputs, targets)
print("Loss:", loss.item())
