import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2 as cv
from DRL.ddpg import decode
from meta_data import train_list, train_list_775
from utils.util import *
from PIL import Image
from torchvision import transforms, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

aug = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomHorizontalFlip(),
     ])

width = 128
convas_area = width * width

color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 0),
         (128, 0, 255), (255, 128, 0), (255, 0, 128)]

img_train_src = []
img_train_tgt = []
img_train_ref = []
img_train_canvas = []
img_train_point = []
img_test_src = []
img_test_tgt = []
img_test_ref = []
img_test_canvas = []
img_test_point = []
train_num = 0
test_num = 0
channels = 3


def stroke_draw(p):
    canvas = np.zeros((128, 128, 3))
    p1 = (p + 1) * 64
    p1 = p1.astype(np.int)
    for i in range(9):
        cv.line(canvas, (p1[i][0], p1[i][1]), (p1[i + 1][0], p1[i + 1][1]), (255, 255, 255), 4)
    for i in range(10):
        cv.circle(canvas, (p1[i][0], p1[i][1]), 2, color[i], -1)
    return canvas


def stroke_normal(p):
    maxx = p[:, 0].max(0)
    maxy = p[:, 1].max(0)
    minx = p[:, 0].min(0)
    miny = p[:, 1].min(0)
    mxy = np.array([(maxx + minx) / 2, (maxy + miny) / 2])
    p = p - mxy
    dxy = p.max() + 1e-9
    p = p / dxy * 0.5
    if p.shape[0] == 1:
        print(p)
    return p


class Paint:
    def __init__(self, batch_size, max_step, font):
        self.batch_size = batch_size
        self.max_step = max_step
        self.test = False
        self.font = font

    def load_data(self):
        global train_num, test_num

        font1 = 'mean'
        font2 = self.font
        print(font1, font2)
        img_list = os.listdir('data/font_concat/{}'.format(font2))
        point1 = np.load('data/font_10_stroke/{}_point.npy'.format(font1))
        point2 = np.load('data/font_10_stroke/{}_point.npy'.format(font2))
        valnum = 0
        for add in img_list:
            img_id = int(add.split('.')[0].split('_')[0])
            stroke = int(add.split('.')[0].split('_')[1])
            stroke -= 1
            if img_id not in train_list_775 and img_id not in train_list[:100]:
                continue
            img = cv.imread('data/font_concat/{}/'.format(font2) + add)
            img = cv.resize(img, (width * 4, width))
            img_ref = img[:, 1 * width:2 * width]
            img_point = np.zeros((width, width, 3))
            img_point[0:10, 0:2, 0] = stroke_normal(point1[img_id][stroke])
            img_point[0:10, 0:2, 1] = stroke_normal(point2[img_id][stroke])
            img_src = stroke_draw(stroke_normal(point1[img_id][stroke]))
            img_tgt = stroke_draw(stroke_normal(point2[img_id][stroke]))
            img_canvas = stroke_draw(stroke_normal(point1[img_id][stroke]))
            img_point[10, 0:2, 0] = img_id
            img_point[10, 0:2, 1] = stroke

            if img_id in train_list_775:
                img_train_src.append(img_src)
                img_train_tgt.append(img_tgt)
                img_train_canvas.append(img_canvas)
                img_train_ref.append(img_ref)
                img_train_point.append(img_point)
                train_num += 1
            elif valnum < 24:
                valnum += 1
                img_test_src.append(img_src)
                img_test_tgt.append(img_tgt)
                img_test_canvas.append(img_canvas)
                img_test_ref.append(img_ref)
                img_test_point.append(img_point)
                test_num += 1

        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))

    def pre_data(self, id, test):
        if test:
            img_src = img_test_src[id]
            img_tgt = img_test_tgt[id]
            img_canvas = img_test_canvas[id]
            img_ref = img_test_ref[id]
            img_point = img_test_point[id]
        else:
            img_src = img_train_src[id]
            img_tgt = img_train_tgt[id]
            img_canvas = img_train_canvas[id]
            img_ref = img_train_ref[id]
            img_point = img_train_point[id]
        return np.transpose(img_src, (2, 0, 1)), np.transpose(img_tgt, (2, 0, 1)), np.transpose(img_canvas, (
            2, 0, 1)), np.transpose(img_ref, (2, 0, 1)), np.transpose(img_point, (2, 0, 1))

    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        self.src = torch.zeros([self.batch_size, channels, width, width], dtype=torch.float).to(device)
        self.tgt = torch.zeros([self.batch_size, channels, width, width], dtype=torch.float).to(device)
        self.canvas = torch.zeros([self.batch_size, channels, width, width], dtype=torch.float).to(device)
        self.ref = torch.zeros([self.batch_size, channels, width, width], dtype=torch.float).to(device)
        self.point = torch.zeros([self.batch_size, channels, width, width], dtype=torch.float).to(device)
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num) % test_num
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.src[i], self.tgt[i], self.canvas[i], self.ref[i], self.point[i] = torch.tensor(
                self.pre_data(id, test)).float()
        self.stepnum = 0
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()

    def observation(self):
        # canvas B * channels * width * width
        # src B * channels * width * width
        # tgt B * channels * width * width
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.float) * self.stepnum
        return torch.cat((self.canvas, self.src, T.to(device), self.tgt, self.ref, self.point),
                         1)  # canvas, src, T, tgt, ref, mask

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)

    def step(self, action):
        self.canvas, self.point[:, 0, 0:10, 0:2] = decode(self.point[:, 0, 0:10, 0:2], action)
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward()  # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), done

    def cal_dis(self):
        return ((self.point[:, 0, 0:10, 0:2] - self.point[:, 1, 0:10, 0:2]) ** 2).sum(2).sqrt().mean(1)

    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
