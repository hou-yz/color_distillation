import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from color_distillation.models.dncnn import DnCNN
from color_distillation.models.unet import UNet


class ColorCNN(nn.Module):
    def __init__(self, in_channel, num_colors, soften=1):
        super().__init__()
        self.num_colors = num_colors
        self.soften = soften
        self.base = UNet(in_channel)
        self.color_mask = nn.Sequential(nn.Conv2d(self.base.out_channel, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, num_colors, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()

    def forward(self, img, training=True):
        B, C, H, W = img.shape
        feat = self.base(img)
        mask = self.color_mask(feat)
        mask = self.mask_softmax(self.soften * mask)
        argmax_mask = torch.argmax(mask, dim=1, keepdim=True)
        argmax_mask = torch.zeros_like(mask).scatter(1, argmax_mask, 1)
        if training:
            weighted_color = (img.unsqueeze(2) * mask.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    mask.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (mask.unsqueeze(1) * weighted_color).sum(dim=2)
        else:
            weighted_color = (img.unsqueeze(2) * argmax_mask.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    argmax_mask.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (argmax_mask.unsqueeze(1) * weighted_color).sum(dim=2)

        return transformed_img, mask


def test():
    img = torch.randn([1, 3, 32, 32])
    model = ColorCNN(3, 128)
    transformed_img = model(img)
    pass

# test()
