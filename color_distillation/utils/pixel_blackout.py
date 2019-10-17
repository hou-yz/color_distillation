import time
import numpy as np
import torch
from PIL import Image
import skimage.segmentation as seg
import skimage.color as color


def update_kmeans_image(dataset, num_colors):
    t0 = time.time()
    for i, img in enumerate(dataset.data):
        image_slic = seg.slic(img, n_segments=num_colors)
        sampled_img = color.label2rgb(image_slic, np.array(img), kind='avg')
        dataset.interpolated_img[i] = torch.from_numpy(sampled_img)
    print(f'kmeans image update, time: {time.time() - t0:.1f}s.')
