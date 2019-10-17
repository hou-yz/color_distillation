import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import *


class GridDownSample(object):
    """ Call PIL functions to first down-sample then up-sample
    Args:
         downsample: The area down sample ratio.
    """

    def __init__(self, downsample=1.0):
        self.hw_downsample = downsample ** 0.5

    def __call__(self, img):
        H, W = img.size
        H_reduced, W_reduced = (np.array([H, W]) * self.hw_downsample).astype(int)
        downsampled_img = img.resize((H_reduced, W_reduced), Image.NEAREST).resize((H, W), Image.LINEAR)
        return downsampled_img

