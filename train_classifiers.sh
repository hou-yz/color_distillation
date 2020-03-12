#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python grid_downsample.py -d cifar10 -a alexnet --train
CUDA_VISIBLE_DEVICES=0 python grid_downsample.py -d cifar100 -a alexnet --train
CUDA_VISIBLE_DEVICES=0 python grid_downsample.py -d stl10 -a alexnet --train
CUDA_VISIBLE_DEVICES=0 python grid_downsample.py -d tiny200 -a alexnet --train
