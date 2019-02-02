import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.conv1 = conv_block(x_dim, hid_dim)
        self.conv2 = conv_block(hid_dim, hid_dim)
        self.conv3 = conv_block(hid_dim, hid_dim)
        self.conv4 = conv_block(hid_dim,z_dim)

    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        feature4 = self.conv4(feature3)
        return feature1.view(feature1.size(0), -1),
               feature2.view(feature2.size(0), -1),
               feature3.view(feature3.size(0), -1),
               feature4.view(feature4.size(0), -1),


