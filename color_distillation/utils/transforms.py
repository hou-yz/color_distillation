from PIL import Image
from io import BytesIO
from color_distillation.utils.dither.palette import Palette
from color_distillation.utils.dither.dithering import error_diffusion_dithering
from torchvision.transforms import *


class MedianCut(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img):
        if self.num_colors is not None:
            if not self.dither:
                sampled_img = img.quantize(colors=self.num_colors, method=0).convert('RGB')
            else:
                palette = Palette(img.quantize(colors=self.num_colors, method=0))
                sampled_img = error_diffusion_dithering(img, palette).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class OCTree(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, method=2).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class KMeans(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

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
