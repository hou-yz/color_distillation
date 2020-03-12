#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`diffusion`
=======================
.. moduleauthor:: hbldh <henrik.blidh@swedwise.com>
Created on 2016-09-12, 11:34
"""

import numpy as np

_DIFFUSION_MAPS = {
    'floyd-steinberg': (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16)
    ),
}


def error_diffusion_dithering(image, palette, method='floyd-steinberg',
                              order=2):
    """Perform image dithering by error diffusion method.
    .. note:: Error diffusion is totally unoptimized and therefore very slow.
        It is included more as a reference implementation than as a useful
        method.
    Reference:
        http://bisqwit.iki.fi/jutut/kuvat/ordered_dither/error_diffusion.txt
    Quantization error of *current* pixel is added to the pixels
    on the right and below according to the formulas below.
    This works nicely for most static pictures, but causes
    an avalanche of jittering artifacts if used in animation.
    Floyd-Steinberg:
              *  7
           3  5  1      / 16
    :param :class:`PIL.Image` image: The image to apply error
        diffusion dithering to.
    :param :class:`~hitherdither.colour.Palette` palette: The palette to use.
    :param str method: The error diffusion map to use.
    :param int order: Metric parameter ``ord`` to send to
        :method:`numpy.linalg.norm`.
    :return: The error diffusion dithered PIL image of type
        "P" using the input palette.
    """
    ni = np.array(image, 'float')

    diff_map = _DIFFUSION_MAPS.get(method.lower())

    for y in range(ni.shape[0]):
        for x in range(ni.shape[1]):
            old_pixel = ni[y, x]
            old_pixel[old_pixel < 0.0] = 0.0
            old_pixel[old_pixel > 255.0] = 255.0
            new_pixel = palette.pixel_closest_colour(old_pixel, order)
            quantization_error = old_pixel - new_pixel
            ni[y, x] = new_pixel
            for dx, dy, diffusion_coefficient in diff_map:
                xn, yn = x + dx, y + dy
                if (0 <= xn < ni.shape[1]) and (0 <= yn < ni.shape[0]):
                    ni[yn, xn] += quantization_error * diffusion_coefficient
    return palette.create_PIL_png_from_rgb_array(np.array(ni, 'uint8'))
