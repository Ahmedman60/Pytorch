import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class CNN(nn.Module):

    def __init__(self, in_channels=3, num_classes=10):

        super(CNN, self).__init__()
        # 3x32x32
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4,
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 1x14x14
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8,
                               kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8*8*8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # same as reshape(x.shape[0],-1) https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        x = self.flatten(x)
        x = self.linear(x)

        return x


# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# hyperparameters
in_ch = 3
classes = 10
lr = 0.001
batch_size = 64
num_epochs = 2
# loading data

train_dataset = datasets.CIFAR10(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Intialize network

model = CNN().to(device)
# loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=lr)


for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device)
        target = target.to(device)

        # forward
        score = model(data)
        loss = criterion(score, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # grident decent step
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


def check_accuracy(model, loader):
    if loader.dataset.train:
        print("checking on the training ")

    else:
        print("Checking on the Test")

    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            score = model(x)
            _, prediction = score.max(1)
            num_correct += (prediction == y).sum()
            num_sample += x.size(0)

        print(
            f"Got {num_correct}/ {num_sample}  with accuracy  {(num_correct/num_sample)*100:.2f}  ")

    model.train()


check_accuracy(model, train_loader)
check_accuracy(model, test_loader)
