import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
