import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
# this will grante that it runs on GPU if it availabe.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFR10  -  Fashion MNIST.
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])


training_data = FashionMNIST(
    root='./', train=True, transform=transforms, download=False)

test_data = FashionMNIST(root='./', train=False,
                         transform=transforms, download=False)

batch_size = 32
training_loader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


def ShowData(images):
    '''
    Images: should be tuple of images and labels (images,labels)
    '''
    figure = plt.figure(figsize=(15, 20))

    row, col = 5, 5

    for i in range(1, row*col+1):
        randomindex = torch.randint(len(images), size=(1,)).item()
        image, label = images[randomindex]

        figure.add_subplot(row, col, i)
        plt.title(labels_map[label], color="Red")
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0), cmap="gray")

    plt.show()


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()

        # input_channels , output channels , kernal_size
        self.conv1 = nn.Conv2d(1, 3, 5)  # 24x24x3
        # this will divide the image dim into half
        self.pool = nn.MaxPool2d(2, 2)  # 12x12x3
        self.conv2 = nn.Conv2d(3, 16, 5)  # 8x8x16
        # 4x4x16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)  # we have 10 classes.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 5

n_total_steps = len(training_loader)  # total Number of batches


for epoch in range(epochs):

    for index, (images, labels) in enumerate(training_loader):

        # moving images and labels to gpu
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index+1) % 10 == 0:

            print(
                f"Epoch {epoch+1}/{epochs} ,Step {index+1}/{n_total_steps},Loss {loss.item():.4f} ")

print("\n Traning is Done. ")


with torch.no_grad():
    # for the whole dataset
    n_correct = 0
    n_samples = 0
    # for each class
    n_classes_correct = [0]*10
    n_classes_samples = [0]*10

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # (batch,10) 10 is the # of classes   is the output (batch,1) -> thenumber and the label i need the label
        _, predicte = torch.max(output, 1)

        n_samples += len(labels)
        n_correct += (predicte == labels).sum().item()

        # for each class now

        for i in range(batch_size):
            if i in labels:
                # labels will be each correct label in specific order [1,3,4,5]
                label = labels[i]
                # predict [1,2,4,2]   the 1 is 1 and 3 is not 2 wrong here and so on
                pred = predicte[i]

                # means we have seen this specific label (count how many times i have seen this specific class in the batch)
                # we can make it using dictionary if we don't know how many classes we have dict.get(label) get it and increase else intialize it then increase
                n_classes_samples[label] += 1

                if label == pred:
                    # label is the indicator for example 1 if it correctly predicted we increase index 1 as so on
                    n_classes_correct[label] += 1

    acc = 100.0 * n_correct/n_samples

    print(f" accuracy of network: {acc:.4f} % ")

    print("Correct classes for each class ", n_classes_correct)
    print("Total sample for each class ", n_classes_samples)

    n_classes_correct = np.array(n_classes_correct)
    n_classes_samples = np.array(n_classes_samples)

    acc = 100.0*(n_classes_correct/n_classes_samples)

    for i in range(10):
        print(f"Class {labels_map[i]} accuracy is {acc[i]:.2f} %")
