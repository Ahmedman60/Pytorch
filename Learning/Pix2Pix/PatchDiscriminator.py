import torch
import torch.nn as nn


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


# Testing the model
model = PatchDiscriminator()
# Batch size = 1, Input channels = 3, Image size = 256x256
input_tensor = torch.randn(1, 3, 256, 256)
output = model(input_tensor)


# class PatchDiscriminator(nn.Module):
#     def __init__(self, input_channels=3):
#         super(PatchDiscriminator, self).__init__()
#         self.model = nn.Sequential(
#             # Input shape: (batch_size, input_channels, 256, 256)
#             nn.Conv2d(input_channels, 64, kernel_size=4,
#                       stride=2, padding=1),  # 1st layer
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2,
#                       padding=1),  # 2nd layer
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4,
#                       stride=2, padding=1),  # 3rd layer
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4,
#                       stride=2, padding=1),  # 4th layer
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, kernel_size=4, stride=1,
#                       padding=1),  # Final layer
#             nn.Sigmoid()  # Output real/fake per patch
#         )

#     def forward(self, x):
#         # Printing the shape of the input after each layer
#         print("Input shape:", x.shape)
#         for layer in self.model:
#             x = layer(x)
#             # print(f"Shape after {layer}: {x.shape}")
#             print(x.shape)
#         return x


# # Testing the model
# model = PatchDiscriminator()
# # Batch size = 1, Input channels = 3, Image size = 256x256
# input_tensor = torch.randn(1, 3, 256, 256)
# output = model(input_tensor)
