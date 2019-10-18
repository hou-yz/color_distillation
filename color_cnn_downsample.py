import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import color_distillation.utils.transforms as T
from color_distillation import models
from color_distillation.models.color_cnn import ColorCNN
from color_distillation.trainer import CNNTrainer
from color_distillation.utils.draw_curve import draw_curve
from color_distillation.utils.logging import Logger
from color_distillation.utils.image_utils import img_color_denormalize, create_coord_map


def main():
    # settings
    parser = argparse.ArgumentParser(description='ColorCNN down sample')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('-a', '--arch', type=str, default='vgg16', choices=models.names())
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--downsample', type=float, default=1.0)
    args = parser.parse_args()

    # seed
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataset
    if args.dataset == 'mnist':
        H, W, C = 28, 28, 1

        train_trans = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,)), ])
        test_trans = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,)), ])
        denormalizer = img_color_denormalize((0.1307,), (0.3081,))

        train_set = datasets.MNIST('./data', train=True, download=True, transform=train_trans)
        test_set = datasets.MNIST('./data', train=False, download=True, transform=test_trans)
    elif args.dataset == 'cifar10':
        H, W, C = 32, 32, 3

        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        test_trans = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        denormalizer = img_color_denormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_trans)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_trans)
    else:
        raise Exception

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    logdir = 'logs/colorcnn/{}/{}/downsample{}/{}'.format(args.dataset, args.arch, args.downsample,
                                                          datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logdir, exist_ok=True)
    copy_tree('./color_distillation', logdir + '/scripts/color_distillation')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    pretrain_cnn = models.create(args.arch, C, 10).cuda()
    pretrain_dir = 'logs/grid/{}/{}/downsample1.0'.format(args.dataset, args.arch) + '/model.pth'
    pretrain_cnn.load_state_dict(torch.load(pretrain_dir))
    pretrain_cnn.eval()
    for param in pretrain_cnn.parameters():
        param.requires_grad = False

    model = ColorCNN(C, int(H * W * args.downsample)).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 40, 2)

    coord_map = create_coord_map([H, W, C], True).cuda()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []

    trainer = CNNTrainer(model, nn.CrossEntropyLoss(), pretrain_cnn, denormalizer, coord_map)

    # learn
    print('Testing...')
    trainer.test(test_loader)

    for epoch in range(1, args.epochs + 1):
        print('Training...')
        train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
        print('Testing...')
        og_test_loss, og_test_prec = trainer.test(test_loader)

        x_epoch.append(epoch)
        train_loss_s.append(train_loss)
        train_prec_s.append(train_prec)
        og_test_loss_s.append(og_test_loss)
        og_test_prec_s.append(og_test_prec)
        draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                   og_test_loss_s, og_test_prec_s)
    # save
    torch.save(model.state_dict(), os.path.join(logdir, 'ColorCNN.pth'))


if __name__ == '__main__':
    main()
