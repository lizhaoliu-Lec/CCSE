import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

from torch.autograd import Variable
import sys

from torchvision.models import resnet18, resnet34, resnet50, resnet152, resnet101


def conv3x3(in_planes, out_planes, stride=1):
    return (nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = (nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.conv2 = (nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.conv3 = (nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet_mover(nn.Module):
    d2resnet = {
        k: v for k, v in zip([18, 34, 50, 101, 152],
                             [resnet18, resnet34, resnet50, resnet101, resnet152])
    }
    d2dim = {
        k: v for k, v in zip([18, 34, 50, 101, 152],
                             [512, 512, 2048, 2048])
    }

    def __init__(self, num_inputs, depth, num_outputs, dropout=0):
        super(ResNet_mover, self).__init__()
        self.in_planes = 64

        resnet = self.d2resnet[depth](pretrained=False)
        dim = self.d2dim[depth]
        self.conv1 = conv3x3(num_inputs, 64, 2)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.dropout = dropout

        self.fc = nn.Linear(dim, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        N, c = x.size(0), x.size(1)
        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape((N, c))
        if self.dropout != 0:
            x = F.dropout(x, self.dropout)
        x = self.fc(x)
        #        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
