import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class CNNTrainer(BaseTrainer):
    def __init__(self, model, criterion, pretrain_model=None, denormalizer=None, coord_map=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.pretrain_model = pretrain_model
        self.denormalizer = denormalizer
        self.coord_map = coord_map
        self.mse_loss = nn.MSELoss()
        if pretrain_model is not None:
            self.has_pretrain = True
        else:
            self.has_pretrain = False

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None, ):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if self.has_pretrain:
                transformed_img = self.model(data, self.coord_map)
                output = self.pretrain_model(transformed_img)
            else:
                output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            if self.has_pretrain:
                loss = self.criterion(output, target) + self.mse_loss(transformed_img, data)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, device=0):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(device), target.cuda(device)
            with torch.no_grad():
                if self.has_pretrain:
                    transformed_img = self.model(data, self.coord_map, training=False)
                    output = self.pretrain_model(transformed_img)
                    # plotting
                    # og_img = self.denormalizer(data[0]).cpu().numpy()
                    # plt.imshow(og_img.reshape([3, 32, 32]).transpose([1, 2, 0]))
                    # plt.show()
                    # downsampled_img = self.denormalizer(transformed_img[0]).cpu().numpy()
                    # plt.imshow(downsampled_img.reshape([3, 32, 32]).transpose([1, 2, 0]))
                    # plt.show()
                else:
                    output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            if self.has_pretrain:
                loss = self.criterion(output, target) + self.mse_loss(transformed_img, data)
            else:
                loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, Prec: {:.1f}%'.format(losses / (len(test_loader) + 1),
                                                         100. * correct / (correct + miss)))

        return losses / len(test_loader), correct / (correct + miss)
