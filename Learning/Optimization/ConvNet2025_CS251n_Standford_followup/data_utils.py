import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=64, num_workers=4):
    """
    Create data loaders for Fashion-MNIST dataset

    Args:
        batch_size (int): Batch size for training and validation
        num_workers (int): Number of workers for data loading

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Fashion-MNIST mean and std
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Load training data
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Load validation data
    val_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def get_class_names():
    """Return Fashion-MNIST class names"""
    return [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]
