# Learning to Structure an Image with Few Colors [[Website](https://hou-yz.github.io/publication/2019-Learning%20to%20Structure%20an%20Image%20with%20Few%20Colors)] [[arXiv](https://arxiv.org/abs/2003.07848)]

```
@article{hou2020learning,
  title={Learning to Structure an Image with Few Colors},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  journal={CVPR},
  year={2020}
}
```


## Overview
We release the PyTorch code for ColorCNN, a newly introduced architecture in our paper *[Learning to Structure an Image with Few Colors](https://hou-yz.github.io/publication/2019-Learning%20to%20Structure%20an%20Image%20with%20Few%20Colors)*.
![system overview](https://hou-yz.github.io/images/ColorCNN_system.png "System overview of image color quantization with ColorCNN.")
 
## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Code](#code)
    * [Training Classifiers](#training-classifiers)
    * [Training & Evaluating ColorCNN](#training-&-evaluating-colorcnn)
    * [Evaluating Traditional Methods](#evaluating-traditional-methods)


## Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.4+ & tochvision
- numpy
- matplotlib
- pillow
- opencv-python

## Data Preparation
By default, all datasets are in `~/Data/`. We use CIFAR10, CIFAR100, STL10, and tiny-imagenet-200 in this project. 
The first three datasets can be automatically downloaded. 

Tiny-imagenet-200 can be downloaded from this [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip). 
Once downloaded, please extract the zip files under `~/Data/tiny200/`. 
Then, run `python color_distillation/utils/tiny_imagenet_val_reformat.py` to reformat the validation set. (thank [@tjmoon0104](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/utils/tiny-imgnet-val-reformat.ipynb) for his code).

Your `~/Data/` folder should look like this
```
Data
├── cifar10/
│   └── ...
├── cifar100/ 
│   └── ...
├── stl10/
│   └── ...
└── tiny200/ 
    ├── train/
    │   └── ...
    ├── val/
    │   ├── n01443537/
    │   └── ...
    └── ...
```

## Code
One can find classifier training & evaluation for traditional color quantization methods in `grid_downsample.py`.
For ColorCNN training & evaluation, please find it in `color_cnn_downsample.py`. 

### Training Classifiers
In order to train classifiers, please specify `'--train'` in the arguments. 
```shell script
python grid_downsample.py -d cifar10 -a alexnet --train
``` 
One can run the shell script `bash train_classifiers.sh` to train AlexNet on all four datasets. 

### Training & Evaluating ColorCNN
Based on the original image pre-trained classifiers, we then train ColorCNN under specific color space sizes. 
```shell script
python color_cnn_downsample.py -d cifar10 -a alexnet --num_colors 2
``` 
Please run the shell script `bash train_test_colorcnn.sh` to train and evaluate *ColorCNN* with AlexNet on all four datasets, under a 1-bit color space. 

### Evaluating Traditional Methods
Based on pre-trained classifiers, one can directly evaluate the performance of tradition color quantization methods. 
```shell script
python python grid_downsample.py -d cifar10 -a alexnet --num_colors 2 --sample_type mcut --dither
``` 
Please run the shell script `bash test_mcut_dither.sh` to evaluate *MedianCut+Dithering* with AlexNet on all four datasets, under a 1-bit color space. 





