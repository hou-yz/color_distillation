import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import skimage.color as color
from io import BytesIO
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
    def __init__(self, num_colors=None):
        self.num_colors = num_colors

    def __call__(self, img):
        if self.num_colors is not None:
            image_slic = seg.slic(img, n_segments=self.num_colors)
            sampled_img = color.label2rgb(image_slic, np.array(img), kind='avg')
            sampled_img = Image.fromarray(sampled_img)
        else:
            sampled_img = img
        return sampled_img


"""
        Convert the image to 'P' mode with the specified number
        of colors.

        :param colors: The desired number of colors, <= 256
        :param method: 0 = median cut
                       1 = maximum coverage
                       2 = fast octree
                       3 = libimagequant
        :param kmeans: Integer
        :param palette: Quantize to the palette of given
                        :py:class:`PIL.Image.Image`.
        :param dither: Dithering method, used when converting from
           mode "RGB" to "P" or from "RGB" or "L" to "1".
           Available methods are NONE or FLOYDSTEINBERG (default).
           Default: 1 (legacy setting)
        :returns: A new image

"""


class MedianCut(object):
    def __init__(self, num_colors=None):
        self.num_colors = num_colors

    def __call__(self, img):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, method=0).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class OCTree(object):
    def __init__(self, num_colors=None):
        self.num_colors = num_colors

    def __call__(self, img):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, method=2).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class KMeans(object):
    def __init__(self, num_colors=None):
        self.num_colors = num_colors

    def __call__(self, img):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, kmeans=2).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class PNGCompression(object):
    def __init__(self, buffer_size_counter):
        self.buffer_size_counter = buffer_size_counter

    def __call__(self, img):
        png_buffer = BytesIO()
        img.save(png_buffer, "PNG")
        self.buffer_size_counter.update(png_buffer.getbuffer().nbytes)
        return img


class JpegCompression(object):
    def __init__(self, buffer_size_counter, quality=5):
        self.buffer_size_counter = buffer_size_counter
        self.quality = quality

    def __call__(self, img):
        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, "JPEG", quality=self.quality)
        self.buffer_size_counter.update(jpeg_buffer.getbuffer().nbytes)
        jpeg_buffer.seek(0)
        sampled_img = Image.open(jpeg_buffer)
        return sampled_img
