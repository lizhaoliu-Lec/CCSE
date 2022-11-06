from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import init


def conv_block(in_channels, output_channels, index, kernel_size=3, BatchNorm=True, relu=True, pooling=True):
    module = nn.Sequential()

    padding = (kernel_size - 1) // 2
    conv = nn.Conv2d(in_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    module.add_module('conv_{}x{}_{}'.format(kernel_size, kernel_size, index), conv)

    if BatchNorm:
        bn = nn.BatchNorm2d(output_channels)
        module.add_module('bn_{0}'.format(index), bn)

    if relu:
        ReLU = nn.LeakyReLU(0.1, inplace=True)
        module.add_module('ReLU_{0}'.format(index), ReLU)

    if pooling:
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        module.add_module('maxpool_{0}'.format(index), pool)

    return module


def conv_block2(in_channels, output_channels, index, kernel_size=3, BatchNorm=True):
    module = nn.Sequential()

    padding = (kernel_size - 1) // 2
    conv = nn.Conv2d(in_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
    module.add_module('conv_{}x{}_{}'.format(kernel_size, kernel_size, index), conv)

    sigmoid = nn.Sigmoid()
    module.add_module('sigmoid_{}'.format(index), sigmoid)

    return module


def deconv_block(in_channels, output_channels, index, kernel_size=3):
    module = nn.Sequential()

    upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    module.add_module('upsample_{0}'.format(index), upsample)

    padding = (kernel_size - 1) // 2
    conv = nn.Conv2d(in_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    module.add_module('conv_{}x{}_{}'.format(kernel_size, kernel_size, index), conv)

    bn = nn.BatchNorm2d(output_channels)
    module.add_module('bn_{0}'.format(index), bn)

    ReLU = nn.LeakyReLU(0.1, inplace=True)
    module.add_module('ReLU_{0}'.format(index), ReLU)

    return module


def build_net():
    module_list = nn.ModuleList()
    # 128x128x3
    block = conv_block(in_channels=3, output_channels=64, kernel_size=3, relu=True, index=0)
    module_list.append(block)
    # 64x64x64
    block = conv_block(in_channels=64, output_channels=128, kernel_size=3, relu=True, index=1)
    module_list.append(block)
    # 32x32x128
    block = conv_block(in_channels=128, output_channels=256, kernel_size=3, relu=True, index=2)
    module_list.append(block)
    # 16x16x256
    block = deconv_block(in_channels=256, output_channels=256, kernel_size=3, index=3)
    module_list.append(block)
    # 32x32x256
    block = deconv_block(in_channels=256 + 128, output_channels=128, kernel_size=3, index=4)
    module_list.append(block)
    # 64x64x128
    block = deconv_block(in_channels=128 + 64, output_channels=64, kernel_size=3, index=5)
    module_list.append(block)
    # 128x128x64
    block = conv_block2(in_channels=64, output_channels=1, index=6)
    module_list.append(block)
    # 128x128x1
    return module_list


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)


class Render(nn.Module):
    def __init__(self):
        super(Render, self).__init__()
        self.module_list = build_net()
        self.module_list.apply(weights_init)

    def forward(self, x):
        output = []
        for iter, module in enumerate(self.module_list):
            x = module(x)
            if iter in [0, 1]:
                output.append(x)
            if iter in [3, 4]:
                x = torch.cat((x, output[4 - iter]), dim=1)
        return x
