import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import ConvNet
from data_utils import get_data_loaders, get_class_names
import datetime


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': running_loss/total,
                         'acc': 100.*correct/total})

    return running_loss/len(train_loader), 100.*correct/total


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss/len(val_loader), 100.*correct/total


def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    name = f"Training_metrics_{datetime.datetime.now()}.png"
    plt.savefig(name)
    plt.close()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    # Initialize model  (use fine_tune only if the model is pretrained in my case i built it from scratch.)
    model = ConvNet(num_classes=10, fine_tune=True).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print('Starting training...')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Fine-tune after 10 epochs
        if epoch == 10:
            print('\nStarting fine-tuning...')
            model.unfreeze_layers()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate/10)

    # Plot metrics
    plot_metrics(train_losses, train_accs, val_losses, val_accs)

    name = f"fashion_mnist_model_{datetime.datetime.now()}.pth"
    # Save model
    torch.save(model.state_dict(), name)
    print('\nTraining completed! Model saved as fashion_mnist_model.pth')


if __name__ == '__main__':
    main()
    # name = f"fashion_mnist_model_{datetime.datetime.now()}.pth"
    # print(name)


'''
#20 epochs with freezing in my opinion it is mistake to do so.

Train Loss: 0.2884, Train Acc: 89.35%
Val Loss: 0.2747, Val Acc: 89.70%
'''

'''
#normally you get higher accuracy but the model not so regualarized.
Train Loss: 0.1200, Train Acc: 95.44%
Val Loss: 0.2084, Val Acc: 92.98%
'''

# if i prove this method i can use to get better results but if i train for longer.
