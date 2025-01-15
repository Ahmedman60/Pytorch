from torchsummary import summary
import torch
import torch.nn as nn
from torchviz import make_dot

# ! important


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.last = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.leaky_relu1(self.conv1(x))
        print("After Conv1:", x.shape)
        x = self.leaky_relu2(self.batch_norm2(self.conv2(x)))
        print("After Conv2:", x.shape)
        x = self.leaky_relu3(self.batch_norm3(self.conv3(x)))
        print("After Conv3:", x.shape)
        x = self.leaky_relu4(self.batch_norm4(self.conv4(x)))
        print("After Conv4:", x.shape)
        x = self.final_conv(x)
        print("After Final Conv:", x.shape)
        x = self.last(x)
        print("After Last Conv:", x.shape)
        x = self.sigmoid(x)
        print("After Final Final Conv:", x.shape)
        return x


# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PatchDiscriminator().to(device)
# Move the input tensor to the same device as the model
input_tensor = torch.randn(1, 3, 256, 256).to(device)

# Run the model
output = model(input_tensor)

summary(model, input_size=(3, 256, 256))

# Create the visualization graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'  # Save the graph as a PNG file
dot.render('patch_discriminator_graph')
