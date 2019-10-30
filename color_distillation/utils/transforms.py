import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import skimage.color as color
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
        downsampled_img = img.resize((H_reduced, W_reduced), Image.NEAREST).resize((H, W), Image.NEAREST)
        return downsampled_img


class SLIC(object):
    """ Call PIL functions to first down-sample then up-sample
    Args:
         downsample: The area down sample ratio.
    """

    def __init__(self, num_colors=None):
        self.num_colors = num_colors

    def __call__(self, img):
        if self.num_colors is not None:
            image_slic = seg.slic(img, n_segments=self.num_colors)
            sampled_img = color.label2rgb(image_slic, np.array(img), kind='avg')
        else:
            sampled_img = img
        return sampled_img
