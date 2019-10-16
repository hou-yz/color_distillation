import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 256
        layers = []
        layers.append(nn.Conv2d(in_channels, features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding))
        self.dncnn = nn.Sequential(*layers)
        self.out_channel = features

    def forward(self, x):
        out = self.dncnn(x)
        return out
