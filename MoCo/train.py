# -*- coding: utf-8 -*-
# gpu_info = !nvidia-smi -i 0
# gpu_info = '\n'.join(gpu_info)
# print(gpu_info)
# nohup python -u mocodataset.py > mocodataset.log 2>&1 &
from datetime import datetime
from functools import partial
import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST,MNIST
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

"""### Set arguments"""

parser = argparse.ArgumentParser(description='Train MoCo on HUST-OBS')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.006, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=24576, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# utils
parser.add_argument('--resume', default='moco1/model_last.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='moco1', type=str, metavar='PATH', help='path to cache (default: none)')
'''
args = parser.parse_args()  # running in command line
'''
# args = parser.parse_args('')  # running in ipynb
args = parser.parse_args()  # running in command line
# set command line arguments here when running in ipynb
args.cos = True
args.schedule = []  # cos in use
args.symmetric = False
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

print(args)

"""### Define data loaders"""

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2
class MiniPair(MNIST):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img=img.numpy()
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2

from torch.utils.data import TensorDataset, DataLoader
class RandomGaussianBlur(object):
    def __init__(self, p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1, max_sigma=1.0):
        self.p = p
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if random.random() < self.p and self.min_kernel_size<self.max_kernel_size:
            kernel_size = random.randrange(self.min_kernel_size, self.max_kernel_size+1, 2)
            sigma = random.uniform(self.min_sigma, self.max_sigma)
            return transforms.functional.gaussian_blur(img, kernel_size, sigma)
        else:
            return img
def jioayan(image):
    if np.random.random() < 0.5:
        image1 = np.array(image)
        # 添加椒盐噪声
        salt_vs_pepper_ratio = np.random.uniform(0.2, 0.4)
        amount = np.random.uniform(0.002, 0.006)
        num_salt = np.ceil(amount * image1.size * salt_vs_pepper_ratio)
        num_pepper = np.ceil(amount * image1.size * (1.0 - salt_vs_pepper_ratio))

        # 在随机位置生成椒盐噪声
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image1.shape]
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image1.shape]
        # image1[coords_salt] = 255
        image1[coords_salt[0], coords_salt[1], :] = 255
        image1[coords_pepper[0], coords_pepper[1], :] = 0
        image = Image.fromarray(image1)
    return image
from torch.utils.data import DataLoader, Dataset
def pengzhang(image):

    # 生成一个0到2之间的随机数
    random_value = random.random() * 3

    if random_value < 1:  # 1/3的概率进行加法操作
        he = random.randint(1, 3)
        kernel = np.ones((he, he), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
    elif random_value < 2:  # 1/3的概率进行除法操作
        he = random.randint(1, 3)  # 生成一个1到10之间的随机整数作为除数
        kernel = np.ones((he,he),np.uint8)
        image = cv2.dilate(image,kernel,iterations = 1)
    return image
class Mydata(Dataset) :
    def __init__(self,transform = None):
        super(Mydata, self).__init__()
        with open('train.json', 'r', encoding='utf8') as f:
            images=json.load(f)
            labels=images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(self.images[item]['path'].replace('\\','/'))
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        x, y = image.size
        if x<15 or y<15:
            m=min(x,y)
            if m%2==0:
                m=m-1
            random_gaussian_blur = RandomGaussianBlur(max_kernel_size=m)
        else:
            random_gaussian_blur = RandomGaussianBlur()
        x, y = 80, 80
        sizey, sizex = 129, 129
        if y < 128:
            while sizey > 128 or sizey < 32:
                sizey = round(random.gauss(y, 30))
        if x < 128:
            while sizex > 128 or sizex < 32:
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
        image1=jioayan(image)
        image2=jioayan(image)
        image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image1 = pengzhang(image1)
        image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        image2 = pengzhang(image2)
        image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        image1 = random_gaussian_blur(image1)
        image2 = random_gaussian_blur(image2)
        train_transform1 = transforms.Compose([
            transforms.Resize((sizey, sizex)),
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.RandomRotation(degrees=(-20, 20), center=(round(64), round(64)), fill=(255, 255, 255)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.84959, 0.84959, 0.84959], [0.30949923, 0.30949923, 0.30949923])])
        im_1 = train_transform1(image1)
        x, y = 80, 80
        sizey, sizex = 129, 129
        if y < 128:
            while sizey > 128 or sizey < 32:
                sizey = round(random.gauss(y, 30))
        if x < 128:
            while sizex > 128 or sizex < 32:
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
        train_transform2 = transforms.Compose([
            transforms.Resize((sizey, sizex)),
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.RandomRotation(degrees=(-20, 20), center=(round(64), round(64)), fill=(255, 255, 255)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.84959, 0.84959, 0.84959], [0.30949923, 0.30949923, 0.30949923])])
        im_2 = train_transform2(image2)
        return im_1, im_2

    def __len__(self):
        return len(self.images)

train_dataset = Mydata()
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 128, num_workers=16,drop_last=True,pin_memory=True,)
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

"""### Define MoCo wrapper"""

class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss

# create model
model = ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
    ).cuda()
print(model.encoder_q)

"""### Define train/test


"""

# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args,trainlist):
    net.train()
    adjust_learning_rate(optimizer, epoch, args,trainlist)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1,im_2 in train_bar:

        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

        loss = net(im_1, im_2)
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.10f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args,trainlist):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

"""### Start training"""

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

# load model if resume
epoch_start = 1
if args.resume is not '':
    checkpoint = torch.load(args.resume, map_location=torch.device('cuda:0'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))

# logging
results = {'train_loss': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

# training loop
trainlist=[]
for epoch in range(epoch_start, args.epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch, args,trainlist)
    results['train_loss'].append(train_loss)
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    # save model
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')