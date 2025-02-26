import torch
print(torch.__version__)          # Check PyTorch version
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)         # Check CUDA version PyTorch uses
print(torch.backends.cudnn.version())  # Check cuDNN version PyTorch uses
