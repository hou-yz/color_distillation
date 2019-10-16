from .alexnet import AlexNet
from .vgg import VGG16

__factory = {
    'alexnet': AlexNet,
    'vgg16': VGG16,
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
