import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class CNNTrainer(BaseTrainer):
    def __init__(self, model, criterion, classifier=None, denormalizer=None, alpha=None, beta=None, gamma=None,
                 visualize=False):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.classifier = classifier
        self.denormalizer = denormalizer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reconsturction_loss = nn.MSELoss()
        self.visualize = visualize
        if classifier is not None:
            self.color_cnn = True
        else:
            self.color_cnn = False

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None, ):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if self.color_cnn:
                transformed_img, mask = self.model(data)
                # regularization
                B, C, H, W = data.shape
                color_max, _ = torch.max(mask.view([B, mask.shape[1], -1]), dim=2)
                color_mean = torch.mean(mask, dim=[2, 3])
                avg_max = torch.mean(color_max)
                std_mean = torch.mean(color_mean.std(dim=1))
                output = self.classifier(transformed_img)
            else:
                output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            if self.color_cnn:
                loss = self.criterion(output, target) + self.alpha * (1 - avg_max) + self.beta * std_mean + \
                       self.gamma * self.reconsturction_loss(data, transformed_img)
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

    def test(self, test_loader):
        buffer_size_counter = 0
        number_of_colors = 0
        dataset_size = 0
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                if self.color_cnn:
                    B, C, H, W = data.shape
                    transformed_img, mask = self.model(data, training=False)
                    output = self.classifier(transformed_img)
                    # image file size
                    M = torch.argmax(mask, dim=1, keepdim=True)  # argmax color index map
                    for i in range(target.shape[0]):
                        number_of_colors += len(M[0].unique())
                        downsampled_img = self.denormalizer(transformed_img[i]).cpu().numpy().squeeze().transpose(
                            [1, 2, 0])
                        downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8'))

                        png_buffer = BytesIO()
                        downsampled_img.save(png_buffer, "PNG")
                        buffer_size_counter += png_buffer.getbuffer().nbytes
                        dataset_size += 1

                    # plotting
                    if self.visualize:
                        plt.imshow(M[11, 0].cpu().numpy(), cmap='Blues')
                        plt.savefig("M.png", bbox_inches='tight')
                        plt.show()
                        og_img = self.denormalizer(data[11]).cpu().numpy().squeeze().transpose([1, 2, 0])
                        plt.imshow(og_img)
                        plt.show()
                        downsampled_img = self.denormalizer(transformed_img[11]).cpu().numpy().squeeze().transpose(
                            [1, 2, 0])
                        plt.imshow(downsampled_img)
                        plt.show()
                        og_img = Image.fromarray((og_img * 255).astype('uint8'))
                        og_img.save('og_img.png')
                        downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8'))
                        downsampled_img.save('downsample_img.png')
                    pass
                else:
                    output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0))
        if self.color_cnn:
            print(f'Average number of colors per image: {number_of_colors / dataset_size}; \n'
                  f'Average image size: {buffer_size_counter / dataset_size:.1f}; '
                  f'Bit per pixel: {buffer_size_counter / dataset_size / H / W:.3f}')

        return losses / len(test_loader), correct / (correct + miss)
