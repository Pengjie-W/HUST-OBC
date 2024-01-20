# -*- coding: utf-8 -*-
import json
import os
import random
from datetime import datetime
import cv2
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
parser = argparse.ArgumentParser(description='Test on HUST-OBS')

parser.add_argument('--lr', '--learning-rate', default=0.015, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')


# utils
parser.add_argument('--resume', default='../model4/model_last.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='test', type=str, metavar='PATH', help='path to cache (default: none)')
args = parser.parse_args()  # running in command line
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
print(args)
args = parser.parse_args()  # running in command line


class RandomGaussianBlur(object):
    def __init__(self, p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1, max_sigma=1.0):
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
        amount = np.random.uniform(0, 0.006)
        num_salt = np.ceil(amount * image1.size / 3 * salt_vs_pepper_ratio)
        num_pepper = np.ceil(amount * image1.size / 3 * (1.0 - salt_vs_pepper_ratio))

        # 在随机位置生成椒盐噪声
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image1.shape]
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image1.shape]
        # image1[coords_salt] = 255
        image1[coords_salt[0], coords_salt[1], :] = 255
        image1[coords_pepper[0], coords_pepper[1], :] = 0
        image = Image.fromarray(image1)
    return image


def pengzhang(image):
    # 生成一个0到2之间的随机数
    random_value = random.random() * 3

    if random_value < 1:  # 1/3的概率进行加法操作
        he = random.randint(1, 3)
        kernel = np.ones((he, he), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
    elif random_value < 2:  # 1/3的概率进行除法操作
        he = random.randint(1, 3)  # 生成一个1到10之间的随机整数作为除数
        kernel = np.ones((he, he), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
    return image


class TrainData(Dataset):
    def __init__(self, transform=None):
        super(TrainData, self).__init__()
        with open('train0.2.json', 'r') as f:
            images = json.load(f)
            labels = images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(self.images[item]['path'].replace('\\','/'))
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        image_width, image_height = image.size
        if image_width > image_height:
            x = 72
            y = round(image_height / image_width * 72)
        else:
            y = 72
            x = round(image_width / image_height * 72)
        sizey, sizex = 129, 129
        if y < 128:
            while sizey > 128 or sizey < 16:
                sizey = round(random.gauss(y, 30))
        if x < 128:
            while sizex > 128 or sizex < 16:
                sizex = round(random.gauss(x, 30))
        dx = 128 - sizex  # 差值
        dy = 128 - sizey
        if dx > 0:
            xl = -1
            while xl > dx or xl < 0:
                xl = round(dx / 2)
                xl = round(random.gauss(xl, 10))
        else:
            xl = 0
        if dy > 0:
            yl = -1
            while yl > dy or yl < 0:
                yl = round(dy / 2)
                yl = round(random.gauss(yl, 10))
        else:
            yl = 0
        yr = dy - yl
        xr = dx - xl
        image = jioayan(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = pengzhang(image)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        random_gaussian_blur = RandomGaussianBlur()
        image = random_gaussian_blur(image)
        train_transform = transforms.Compose([
            transforms.Resize((sizey,sizex)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.RandomRotation(degrees=(-15, 15), center=(round(64), round(64)), fill=(255, 255, 255)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.85561067, 0.8557507, 0.85504097], [0.30982512, 0.30969524, 0.31024888])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label

    def __len__(self):
        return len(self.images)


class TestData(Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        with open('test0.2.json', 'r') as f:
            images = json.load(f)
            labels = images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(self.images[item]['path'].replace('\\','/'))
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        width, height = image.size
        if width>height:
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
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.85561067, 0.8557507, 0.85504097], [0.30982512, 0.30969524, 0.31024888])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label,self.images[item]['path'].replace('\\','/')

    def __len__(self):
        return len(self.images)


train_dataset = TrainData()
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=0, pin_memory=True)


test_dataset = TestData()
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=128, num_workers=16, pin_memory=True)


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
                    nn.Flatten(), nn.Linear(2048, 1597))

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
        # total_num += data_loader.abatch_size
        total_num += image.shape[0]
        total_loss += l.item() * image.shape[0]
        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, trainacc: {:.6f}'.format(epoch, args.epochs,
                                                                                      optimizer.param_groups[0]['lr'],
                                                                                      total_loss / total_num,
                                                                                      trainacc / total_num))

    return total_loss / total_num, trainacc / total_num

def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # 如果没有找到匹配的键，返回None或其他适当的值
with open('label扣0.2.json', 'r') as f:
    data = json.load(f)

def test(net, test_data_loader, epoch, args):
    net.eval()
    pathlist = []
    labellist = []
    truelabel=[]
    testacc, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(test_data_loader)
    with torch.no_grad():
        for image, label,path in test_bar:
            image, label = image.cuda(0), label.cuda(0)
            y_hat = net(image)
            y_hat = torch.argmax(y_hat, dim=1)
            truelabel+=[int(x) for x in label.tolist()]
            label = [int(x) for x in y_hat.tolist()]
            labellist=labellist+label
            path=list(path)
            pathlist=pathlist+path
        dataset={}
        num=0
        for i in range(len(pathlist)):
            if labellist[i]!=truelabel[i]:
                path = pathlist[i]
                label=find_key_by_value(data,labellist[i])
                dataset[path]=label
                num+=1
        print(num/len(pathlist))
        with open('错误结果.json', 'w') as f:
            json.dump(dataset, f, ensure_ascii=False)
    return testacc / total_num

results = {'train_loss': [], 'train_acc': [],'test_acc': [], 'lr': []}
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    net.apply(init_weights)

for epoch in range(epoch_start, args.epochs + 1):
    test_acc = test(net, test_loader, epoch, args)
    results['test_acc'].append(test_acc)
    results['lr'].append(args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)))
