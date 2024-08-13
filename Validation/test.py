# -*- coding: utf-8 -*-
import json
import os
import random
from datetime import datetime
import cv2
import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
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
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')


# utils
parser.add_argument('--resume', default='./max_val_acc.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='test', type=str, metavar='PATH', help='path to cache (default: none)')
args = parser.parse_args()  # running in command line
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
print(args)
args = parser.parse_args()  # running in command line

class TestData(Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        with open('Validation_test.json', 'r', encoding='utf8') as f:
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
            transforms.Normalize([0.85233593, 0.85246795, 0.8517555], [0.31232414, 0.3122127, 0.31273854])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label,self.images[item]['path'].replace('\\','/')

    def __len__(self):
        return len(self.images)



test_dataset = TestData()
test_loader = DataLoader(test_dataset, shuffle=True,  batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True)


net = torchvision.models.resnet50(pretrained=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 1588)
net = net.cuda(0)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
loss = nn.CrossEntropyLoss()





def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    cmp = torch.eq(y_hat, y)
    return float(torch.sum(cmp).item())



def test(net, test_data_loader, epoch, args):
    net.eval()
    all_labels = []
    all_preds = []
    testacc, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(test_data_loader)
    with torch.no_grad():
        for image, label,path in test_bar:
            image, label = image.cuda(0), label.cuda(0)
            y_hat = net(image)
            _, preds = torch.max(y_hat, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_num += image.shape[0]
            testacc += accuracy(y_hat, label)
            test_bar.set_description(
                'Test Epoch: [{}/{}], testacc: {:.6f}'.format(epoch, args.epochs, testacc / total_num))
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    print(f'Macro-averaged F1 score: {f1_macro}')
    print(f'Micro-averaged F1 score: {f1_micro}')
    return testacc / total_num

results = {'train_loss': [], 'train_acc': [],'test_acc': [], 'lr': []}
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch']
    print('Loaded from: {}'.format(args.resume))
else:
    net.apply(init_weights)

test_acc = test(net, test_loader, epoch_start, args)
print(test_acc)