#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python color_cnn_downsample.py -d cifar10 -a alexnet --num_colors 2
CUDA_VISIBLE_DEVICES=0 python color_cnn_downsample.py -d cifar100 -a alexnet --num_colors 2
CUDA_VISIBLE_DEVICES=0 python color_cnn_downsample.py -d stl10 -a alexnet --num_colors 2
CUDA_VISIBLE_DEVICES=0 python color_cnn_downsample.py -d tiny200 -a alexnet --num_colors 2
