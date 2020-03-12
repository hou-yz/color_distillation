import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


class img_color_denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1]).cuda()
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1]).cuda()

    def __call__(self, tensor):
        return tensor * self.std + self.mean


class img_color_normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1]).cuda()
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1]).cuda()

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def create_coord_map(img_size, with_r=False):
    H, W, C = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, rr], dim=1)
    return ret
