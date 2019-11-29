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
from color_distillation.loss.label_smooth import LSR_loss
from color_distillation.models.color_cnn import ColorCNN
from color_distillation.trainer import CNNTrainer
from color_distillation.utils.load_checkpoint import checkpoint_loader
from color_distillation.utils.draw_curve import draw_curve
from color_distillation.utils.logger import Logger
from color_distillation.utils.image_utils import img_color_denormalize


def main():
    # settings
    parser = argparse.ArgumentParser(description='ColorCNN down sample')
    parser.add_argument('--num_colors', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1, help='multiplier of regularization terms')
    parser.add_argument('--beta', type=float, default=0, help='multiplier of regularization terms')
    parser.add_argument('--gamma', type=float, default=0, help='multiplier of reconstruction loss')
    parser.add_argument('--color_jitter', type=float, default=1)
    parser.add_argument('--color_norm', type=float, default=4, help='normalizer for color palette')
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--soften', type=float, default=1, help='soften coefficient for softmax')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200'])
    parser.add_argument('-a', '--arch', type=str, default='vgg16', choices=models.names())
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'dncnn'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train_classifier', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    args = parser.parse_args()

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    data_path = os.path.expanduser('~/Data/') + args.dataset
    if args.dataset == 'svhn':
        H, W, C = 32, 32, 3
        num_class = 10

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.SVHN(data_path, split='train', download=True, transform=train_trans)
        test_set = datasets.SVHN(data_path, split='test', download=True, transform=test_trans)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        H, W, C = 32, 32, 3
        num_class = 10 if args.dataset == 'cifar10' else 100

        normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_trans)
        else:
            train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=test_trans)
    elif args.dataset == 'imagenet':
        H, W, C = 224, 224, 3
        num_class = 1000

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageNet(data_path, split='train', transform=train_trans)
        test_set = datasets.ImageNet(data_path, split='val', transform=test_trans)
    elif args.dataset == 'stl10':
        H, W, C = 96, 96, 3
        num_class = 10
        # smaller batch size
        args.batch_size = 32

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(96, padding=12), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.STL10(data_path, split='train', download=True, transform=train_trans)
        test_set = datasets.STL10(data_path, split='test', download=True, transform=test_trans)
    elif args.dataset == 'tiny200':
        H, W, C = 64, 64, 3
        num_class = 200

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    else:
        raise Exception

    # network specific setting
    if args.arch == 'alexnet':
        if 'cifar' not in args.dataset:
            args.color_norm = 1

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    logdir = 'logs/colorcnn/{}/{}/{}colors/{}'.format(args.dataset, args.arch,
                                                      'full_' if args.num_colors is None else args.num_colors,
                                                      datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.resume is None:
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
    classifier = models.create(args.arch, num_class, not args.train_classifier).cuda()
    if not args.train_classifier:
        if args.dataset != 'imagenet':
            resume_fname = 'logs/grid/{}/{}/full_colors'.format(args.dataset, args.arch) + '/model.pth'
            classifier.load_state_dict(torch.load(resume_fname))
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False

    model = ColorCNN(args.backbone, args.num_colors, args.soften, args.color_norm, args.color_jitter).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
    #                                                 steps_per_epoch=len(train_loader), epochs=args.epochs)

    # loss
    if args.label_smooth:
        criterion = LSR_loss(args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []

    trainer = CNNTrainer(model, criterion, args.num_colors, classifier, denormalizer, args.alpha, args.beta, args.gamma)

    # learn
    if args.resume is None:
        # print('Testing...')
        # trainer.test(test_loader)

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
    else:
        resume_dir = 'logs/colorcnn/{}/{}/{}colors/'.format(
            args.dataset, args.arch, 'full_' if args.num_colors is None else args.num_colors) + args.resume
        resume_fname = resume_dir + '/ColorCNN.pth'
        model.load_state_dict(torch.load(resume_fname))
        model.eval()
        print('Test loaded model...')
        trainer.test(test_loader, args.visualize)


if __name__ == '__main__':
    main()
