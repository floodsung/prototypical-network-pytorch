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
        self.fc = nn.Linear(1600,512)
        self.out_channels = 512

    def forward(self, x, M):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) #[B,1600]

        x = torch.cat([torch.rand(x.shape[0]*M,1600),x.repeat(1,M).view(-1, 1600)],dim=1)

        x = self.fc(x)
        return x.view(-1,M,512)

