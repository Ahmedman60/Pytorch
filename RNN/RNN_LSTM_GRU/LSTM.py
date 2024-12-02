import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters  (we think of Nx1x28x28  as 28 sequence each with 28 features)

classes = [str(i) for i in range(10)]
input_size = 28
sequence_lenght = 28
num_layers = 2
hidden_size = 256
learning_rate = 0.001
num_classes = 10
batch_size = 64
num_epochs = 2
direction = 1
# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Creating RNN


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, direction, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = direction
        # hidden_size are the numbers of Nodes in each layer
        # Batchsize,sequencesize,featues
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers, batch_first=True, bidirectional=(self.direction == 2))
        # i have modified the code to include bidirection using one variable to control my direction. if direction=1  it will be false
        # you just pass the input size and you can pass any sequence lenghth all sequences here have same amout 28
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*self.direction, x.size(0),
                         self.hidden_size).to(device)  # since we have one direction num_layer only
        c0 = torch.zeros(self.num_layers*self.direction, x.size(0),
                         self.hidden_size).to(device)

        # out: tensor of shape (batch_size, seq_length, hidden_size)  #x should be in shape batch,seq,featues
        # the _ is the hidden state here i don't use it or store it.
        out, _ = self.rnn(x, (h0, c0))  # from document it take x and h0
        # Decode the hidden state of the last time step
        # the output will be in (Batch,sequence,hiddensize)
        # (N,28,256)
        # out = out.reshape(out.shape[0], -1) this takes all i need only last hidden
        out = out[:, -1, :]
        # The output will be (N,hiddensize) (N,256)
        out = self.linear(out)
        out = self.softmax(out)
        return out


# we you call the object of RNN it automatically call forward function so remember this.
# creating object is a thing and calling the object is another thing.
model = RNN(input_size, hidden_size, num_layers,
            direction, num_classes).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def Train(device, input_size, sequence_lenght, num_epochs, train_loader, model, criterion, optimizer):
    all_losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for i, (images, labels) in enumerate(train_loader):

            # origin shape: [N, 1, 28, 28]
            # resized shape: [N, 28, 28]
            images = images.reshape(-1, sequence_lenght, input_size).to(device)
            labels = labels.to(device)

        # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())  # loss of each sequence
        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        all_losses.extend(epoch_losses)
    return all_losses
# print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluation Loop


def Evaluation(device, input_size, sequence_lenght, test_loader, model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_lenght, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
        # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')
        torch.save(model.state_dict(), 'model.ckpt')


def plot_losses(losses):
    """
    Plot the training losses over batches.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.grid()
    plt.show()


def display_and_predict_example(model, dataset, device, sequence_length, input_size, classes):
    """
    Display an example image and predict its class.

    Args:
        model: The trained RNN model.
        dataset: The dataset from which to sample an image.
        device: The device (CPU or GPU) on which the model is running.
        sequence_length: The sequence length of the input (28 for MNIST).
        input_size: The input size of the RNN (28 for MNIST).
        classes: List of class names (digits 0-9 for MNIST).
    """
    # Set model to evaluation mode
    model.eval()  # no need for it i don't have dropout but incase..

    # Randomly select an image and its label
    index = torch.randint(len(dataset), (1,)).item()
    image, label = dataset[index]

    # Display the image
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"True Label: {classes[label]}")
    plt.axis("off")
    plt.show()

    # Prepare the image for the model
    image = image.reshape(1, sequence_length, input_size).to(device)

    # Predict the class
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = classes[predicted.item()]

    print(f"Predicted Class: {predicted_label}")


# Train
losess = Train(device, input_size, sequence_lenght, num_epochs,
               train_loader, model, criterion, optimizer)
plot_losses(losess)

# Test
Evaluation(device, input_size, sequence_lenght, test_loader, model)


# Use the function with the test dataset
display_and_predict_example(
    model, test_dataset, device, sequence_lenght, input_size, classes)

# https: // pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial
# https://www.youtube.com/watch?v=0_PgWWmauHk&t=239s&ab_channel=PatrickLoeber


# LSTM was best one

Evaluation()
