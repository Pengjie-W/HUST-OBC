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
# utils
parser.add_argument('--resume', default='model_last.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='test', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('--k', default=20, type=int)
args = parser.parse_args()  # running in command line

if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
print(args)
args = parser.parse_args()  # running in command line

with open('label.json', 'r', encoding='utf8') as f:
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
        image = Image.open(self.images[item])
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        # 获取当前图像的尺寸
        width, height = image.size
        if width > height:
            dy = width - height

            yl = round(dy / 2)
            yr = dy - yl
            train_transform = transforms.Compose([
                transforms.Pad([0, yl, 0, yr], fill=(255, 255, 255), padding_mode='constant'),
            ])
        else:
            dx = height - width
            xl = round(dx / 2)
            xr = dx - xl
            train_transform = transforms.Compose([
                transforms.Pad([xl, 0, xr, 0], fill=(255, 255, 255), padding_mode='constant'),
            ])

        image = train_transform(image)
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.7760929, 0.7760929, 0.7760929], [0.39767382, 0.39767382, 0.39767382])])
        image = train_transform(image)
        return image,self.images[item]

    def __len__(self):
        return len(self.images)

test_dataset = TestData()
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=0, pin_memory=True)


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


def test(net, test_data_loader):
    net.eval()
    testacc, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(test_data_loader)
    with torch.no_grad():
        pathlist=[]
        labellist=[]
        for image,path in test_bar:
            image = image.cuda(0)
            y_hat = net(image)
            # y_hat = torch.argmax(y_hat, dim=1)
            y_hat = torch.topk(y_hat, args.k, dim=1)[1]
            label = y_hat.tolist()
            labellist=labellist+label
            path=list(path)
            pathlist=pathlist+path

        dataset={}
        for i in range(len(pathlist)):
            path_label=[]
            path=pathlist[i]
            label=labellist[i]
            for j in label:
                j=str(j)
                path_label.append(data[j])
            dataset[path]=path_label
        with open('result.json', 'w',encoding='utf8') as f:
            json.dump(dataset, f, ensure_ascii=False)
    return


results = {'train_loss': [], 'train_acc': [], 'lr': []}
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    net.apply(init_weights)

test_acc = test(net, test_loader)