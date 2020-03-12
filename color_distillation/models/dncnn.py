import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels, num_of_layers=17, feature_dim=64):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels, feature_dim, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(feature_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.out_channel = feature_dim

    def forward(self, x):
        out = self.dncnn(x)
        return out
