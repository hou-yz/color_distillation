from .alexnet import AlexNet
from .vgg import VGG16
from .resnet import ResNet18, ResNet50, ResNet152
from .mobilenetv2 import MobileNetV2

__factory = {
    'alexnet': AlexNet,
    'vgg16': VGG16,
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet152': ResNet152,
    'mobilenetv2': MobileNetV2,
}


def names():
    return sorted(__factory.keys())


def create(name, in_channel, out_channel):
    """
    Create a model instance.
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](in_channel, out_channel)
