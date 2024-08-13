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
parser = argparse.ArgumentParser(description='Train on HUST-OBC')

parser.add_argument('--lr', '--learning-rate', default=0.015, type=float, metavar='LR', help='initi'
                                                                                             'al learning rate',
                    dest='lr')
parser.add_argument('--epochs', default=600, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='output', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('--checkpoint_freq', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()  # running in command line
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
print(args)
args = parser.parse_args()  # running in command line
seed=args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
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

# nohup python train_new.py > output.log 2>&1 &^C
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
        with open('Validation_train.json', 'r',encoding='utf8') as f:
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
        # x, y = 72,72
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
            transforms.Normalize([0.85233593, 0.85246795, 0.8517555], [0.31232414, 0.3122127, 0.31273854])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label

    def __len__(self):
        return len(self.images)


class ValData(Dataset):
    def __init__(self, transform=None):
        super(ValData, self).__init__()
        with open('Validation_val.json', 'r',encoding='utf8') as f:
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
        return image, label

    def __len__(self):
        return len(self.images)


train_dataset = TrainData()
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True)


val_dataset = ValData()
val_loader = DataLoader(val_dataset, shuffle=True, batch_size = args.batch_size, num_workers=args.num_workers, pin_memory=True)


net = torchvision.models.resnet50(pretrained=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 1588)
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
    total_loss, total_num, trainacc, train_bar = 0.0, 0, 0.0, tqdm(data_loader,ncols=100)
    all_labels = []
    all_preds = []
    for image, label in train_bar:
        image, label = image.cuda(0), label.cuda(0)
        label = label.long()
        y_hat = net(image)
        _, preds = torch.max(y_hat, 1)
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
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
    # 计算 F1 分数
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    # print(f'Macro-averaged F1 score: {f1_macro}')
    # print(f'Micro-averaged F1 score: {f1_micro}')

    return total_loss / total_num, trainacc / total_num,f1_macro,f1_micro


def val(net, val_data_loader, epoch, args):
    net.eval()
    valacc, total_top5, total_num, val_bar = 0.0, 0.0, 0, tqdm(val_data_loader)
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for image, label in val_bar:
            image, label = image.cuda(0), label.cuda(0)
            y_hat = net(image)
            _, preds = torch.max(y_hat, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_num+=image.shape[0]
            valacc += accuracy(y_hat, label)
            val_bar.set_description(
                'Val Epoch: [{}/{}], valacc: {:.6f}'.format(epoch, args.epochs, valacc / total_num))
    # 计算 F1 分数
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    # print(f'Macro-averaged F1 score: {f1_macro}')
    # print(f'Micro-averaged F1 score: {f1_micro}')
    return valacc / total_num,f1_macro,f1_micro


# results = {'train_loss': [], 'train_acc': [], 'val_acc': []}
results = {'train_loss': [], 'train_acc': [],'train_f1_macro':[],'train_f1_micro':[],'val_acc': [],'val_f1_macro':[],'val_f1_micro':[], 'lr': []}
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    net.apply(init_weights)

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
max_val_acc=0
max_val_f1_macro=0
max_val_f1_micro=0
for epoch in range(epoch_start, args.epochs + 1):
    train_loss, train_acc,train_f1_macro,train_f1_micro = train(net, train_loader, optimizer, epoch, args)
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['train_f1_macro'].append(train_f1_macro)
    results['train_f1_micro'].append(train_f1_micro)
    val_acc,val_f1_macro,val_f1_micro = val(net, val_loader, epoch, args)
    results['val_acc'].append(val_acc)
    results['val_f1_macro'].append(val_f1_macro)
    results['val_f1_micro'].append(val_f1_micro)
    results['lr'].append(args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    if (epoch) % args.checkpoint_freq == 0:
        checkpoint_name = f'checkpoint_ep{epoch:04}.pth'
        # save model
        torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), },
                   args.results_dir + '/'+checkpoint_name)
    if epoch>300 and max_val_acc<val_acc:
        max_val_acc=val_acc
        checkpoint_name = f'max_val_acc.pth'
        # save model
        torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), },
                   args.results_dir + '/' + checkpoint_name)
    if epoch>300 and max_val_f1_macro<val_f1_macro:
        max_val_f1_macro=val_f1_macro
        checkpoint_name = f'max_val_f1_macro.pth'
        # save model
        torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), },
                   args.results_dir + '/' + checkpoint_name)
    if epoch>300 and max_val_f1_micro<val_f1_micro:
        max_val_f1_micro=val_f1_micro
        checkpoint_name = f'max_val_f1_micro.pth'
        # save model
        torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), },
                   args.results_dir + '/' + checkpoint_name)