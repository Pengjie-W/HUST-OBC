# -*- coding: utf-8 -*-
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import math
import argparse
from tqdm import tqdm
import pandas as pd

"""### Set arguments"""

parser = argparse.ArgumentParser(description='Test on Chinese OCR')

parser.add_argument('--lr', '--learning-rate', default=0.015, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--epochs', default=601, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
# utils
parser.add_argument('--resume', default='model3/model_last.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='test', type=str, metavar='PATH', help='path to cache (default: none)')
args = parser.parse_args()  # running in command line
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
print(args)
args = parser.parse_args()  # running in command line


class RandomGaussianBlur(object):
    def __init__(self, p=0.5, min_kernel_size=3, max_kernel_size=21, min_sigma=0.1, max_sigma=1.6):
        self.p = p
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if random.random() < self.p and self.min_kernel_size < self.max_kernel_size:
            kernel_size = random.randrange(self.min_kernel_size, self.max_kernel_size + 1, 2)
            sigma = random.uniform(self.min_sigma, self.max_sigma)
            return transforms.functional.gaussian_blur(img, kernel_size, sigma)
        else:
            return img

def jioayan(image):
    if np.random.random() < 0.5:
        image1 = np.array(image)
        # 添加椒盐噪声
        salt_vs_pepper_ratio = np.random.uniform(0, 0.4)
        amount = np.random.uniform(0, 0.01)
        num_salt = np.ceil(amount * image1.size / 3 * salt_vs_pepper_ratio)
        num_pepper = np.ceil(amount * image1.size / 3 * (1.0 - salt_vs_pepper_ratio))

        # 在随机位置生成椒盐噪声
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image1.shape]
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image1.shape]
        image1[coords_salt[0], coords_salt[1], :] = 255
        image1[coords_pepper[0], coords_pepper[1], :] = 0
        image = Image.fromarray(image1)
    return image
class TrainData(Dataset):
    def __init__(self, transform=None):
        super(TrainData, self).__init__()
        with open('训练集.json', 'r') as f:
            images = json.load(f)
            labels = images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(self.images[item]['path'])
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        x, y = 64,64
        sizey, sizex = 129, 129
        if y < 128:
            while sizey > 128 or sizey <= 0:
                sizey = round(random.gauss(y, 30))
        if x < 128:
            while sizex > 128 or sizex <= 0:
                sizex = round(random.gauss(x, 30))
        dx = 128 - sizex  # 差值
        dy = 128 - sizey
        xl = 0
        yl = 0
        if dx > 1:
            while xl >= dx or xl <= 0:
                xl = round(dx / 2)
                xl = round(random.gauss(xl, 10))
        if dy > 1:
            while yl >= dy or yl <= 0:
                yl = round(dy / 2)
                yl = round(random.gauss(yl, 10))
        yr = dy - yl
        xr = dx - xl
        image = jioayan(image)
        random_gaussian_blur = RandomGaussianBlur()
        image = random_gaussian_blur(image)
        train_transform = transforms.Compose([
            transforms.Resize((sizey, sizex)),
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.RandomRotation(degrees=(-15, 15), center=(round(64), round(64)), fill=(255, 255, 255)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.7814565, 0.7814565, 0.7814565], [0.3689568, 0.3689568, 0.3689568])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label

    def __len__(self):
        return len(self.images)
with open('label对应表.json', 'r', encoding='utf8') as f:
    data = json.load(f)
class TestData(Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        with open('use.json', 'r') as f:
            images = json.load(f)
            labels = images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        # image = Image.open(self.images[item]['path'])
        image = Image.open(self.images[item])
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        # 获取当前图像的尺寸
        width, height = image.size

        # 缩小图像到一半尺寸
        image = image.resize((width // 2, height // 2))
        dx = 128 - width // 2  # 差值
        dy = 128 - height // 2
        xl = round(dx / 2)
        yl = round(dy / 2)
        yr = dy - yl
        xr = dx - xl
        train_transform = transforms.Compose([
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize([0.7760929, 0.7760929, 0.7760929], [0.39767382, 0.39767382, 0.39767382])])
        image = train_transform(image)
        return image,self.images[item]

    def __len__(self):
        return len(self.images)

test_dataset = TestData()
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=16, pin_memory=True)


class Residual(nn.Module):
    def __init__(self, input_channels, min_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, min_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(min_channels, min_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.Conv2d(min_channels, num_channels,
                               kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(min_channels)
        self.bn2 = nn.BatchNorm2d(min_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, min_channels, num_channels, num_residuals, stride,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, min_channels, num_channels,
                                use_1x1conv=True, strides=stride))
        elif first_block and i == 0:
            blk.append(Residual(input_channels, min_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, min_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 256, 3, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(256, 128, 512, 4, 2))
b4 = nn.Sequential(*resnet_block(512, 256, 1024, 6, 2))
b5 = nn.Sequential(*resnet_block(1024, 512, 2048, 2, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(2048, 88899))

net = net.cuda(0)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
loss = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    cmp = torch.eq(y_hat, y)
    return float(torch.sum(cmp).item())


def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)
    total_loss, total_num, trainacc, train_bar = 0.0, 0, 0.0, tqdm(data_loader)
    for image, label in train_bar:
        image, label = image.cuda(0), label.cuda(0)

        y_hat = net(image)

        train_optimizer.zero_grad()
        l = loss(y_hat, label)
        l.backward()
        train_optimizer.step()
        trainacc += accuracy(y_hat, label)
        total_num += data_loader.batch_size
        total_loss += l.item() * data_loader.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, trainacc: {:.6f}'.format(epoch, args.epochs,
                                                                                      optimizer.param_groups[0]['lr'],
                                                                                      total_loss / total_num,
                                                                                      trainacc / total_num))

    return total_loss / total_num, trainacc / total_num


def test(net, test_data_loader):
    net.eval()
    testacc, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(test_data_loader)
    with torch.no_grad():
        pathlist=[]
        labellist=[]
        for image,path in test_bar:
            image = image.cuda(0)
            y_hat = net(image)
            y_hat = torch.argmax(y_hat, dim=1)
            label = [int(x) for x in y_hat.tolist()]
            labellist=labellist+label
            path=list(path)
            pathlist=pathlist+path

        dataset={}
        for i in range(len(pathlist)):
            path=pathlist[i]
            label=str(labellist[i])
            label=data[label]
            dataset[path]=label
        with open('result.json', 'w') as f:
            json.dump(dataset, f, ensure_ascii=False)
    return


results = {'train_loss': [], 'train_acc': [], 'lr': []}
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    net.apply(init_weights)

test_acc = test(net, test_loader)