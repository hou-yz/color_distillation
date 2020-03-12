#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python grid_downsample.py -j 8 -d cifar10 --num_colors 2 -a alexnet --sample_type mcut --dither
CUDA_VISIBLE_DEVICES=3 python grid_downsample.py -j 8 -d cifar100 --num_colors 2 -a alexnet --sample_type mcut --dither
CUDA_VISIBLE_DEVICES=3 python grid_downsample.py -j 8 -d stl10 --num_colors 2 -a alexnet --sample_type mcut --dither
CUDA_VISIBLE_DEVICES=3 python grid_downsample.py -j 8 -d tiny200 --num_colors 2 -a alexnet --sample_type mcut --dither