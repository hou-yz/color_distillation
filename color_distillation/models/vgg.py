'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, out_channel):
        super(VGG, self).__init__()
        self.features = self._make_layers(3, cfg[vgg_name])
        self.features[-1] = nn.Sequential()  # remove last pooling layer
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.has_classifier = out_channel != 0
        if self.has_classifier:
            self.classifier = nn.Linear(512, out_channel)
        else:
            self.out_channel = 512

    def forward(self, x):
        out = self.features(x)
        out = self.global_average_pooling(out)
        out = out.view(out.size(0), -1)
        if self.has_classifier:
            out = self.classifier(out)
        return out

    def _make_layers(self, in_channel, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channel, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channel = x
        return nn.Sequential(*layers)


def VGG16(out_channel):
    return VGG('VGG16', out_channel)


def test():
    net = VGG('VGG11', 10)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
