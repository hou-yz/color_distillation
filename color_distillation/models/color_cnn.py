import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from color_distillation.models.dncnn import DnCNN
from color_distillation.models.unet import UNet


class ColorCNN(nn.Module):
    def __init__(self, in_channel, num_colors):
        super().__init__()
        self.num_colors = num_colors
        self.base = UNet(in_channel)
        self.color_mask = nn.Sequential(nn.Conv2d(self.base.out_channel, 512, 1), nn.ReLU(),
                                        nn.Conv2d(512, num_colors, 1))
        self.mask_softmax = nn.Softmax2d()

    def forward(self, img, coord_map, training=True):
        feat = self.base(img)
        mask = self.color_mask(feat)
        mask = self.mask_softmax(mask)
        argmax_mask = torch.argmax(mask, dim=1, keepdim=True)
        argmax_mask = torch.zeros_like(mask).scatter(1, argmax_mask, 1)
        if training:
            weighted_color = (img.unsqueeze(2) * mask.unsqueeze(1)).mean(dim=[3, 4], keepdim=True)
            transformed_img = (mask.unsqueeze(1) * weighted_color).sum(dim=2)
        else:
            weighted_color = (img.unsqueeze(2) * argmax_mask.unsqueeze(1)).mean(dim=[3, 4], keepdim=True)
            transformed_img = (argmax_mask.unsqueeze(1) * weighted_color).sum(dim=2)

        # regularization
        B, C, H, W = img.shape
        mean_max, _ = torch.max(mask.view([B, self.num_colors, -1]), dim=2)
        mean_max = torch.mean(mean_max)
        std_mean = torch.mean(torch.mean(mask, dim=[2, 3]).std(dim=1))
        return transformed_img, mean_max, std_mean


def test():
    img = torch.randn([1, 3, 32, 32])
    model = ColorCNN(3, 128)
    transformed_img = model(img)
    pass

# test()
