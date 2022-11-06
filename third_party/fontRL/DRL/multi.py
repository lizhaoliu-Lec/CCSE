import cv2
import torch
import numpy as np
from env import Paint
from utils.util import *
from DRL.ddpg import decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fastenv():
    def __init__(self, max_episode_length=10, env_batch=64, writer=None, font=''):
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = Paint(self.env_batch, self.max_episode_length, font)
        self.env.load_data()
        self.writer = writer
        self.test = False
        self.log = 0

    def save_image(self, log, step):
        for i in range(self.env_batch):
            if self.env.imgid[i] <= 10:
                # canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                canvas = to_numpy(self.env.canvas[i].permute(1, 2, 0))  # .repeat(3, axis=2)
                self.writer.add_image('{}/canvas_{}.png'.format(str(self.env.imgid[i]), str(step)), canvas, log)
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if self.env.imgid[i] < 50:
                    # tgt = cv2.cvtColor((to_numpy(self.env.tgt[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    # src = cv2.cvtColor((to_numpy(self.env.src[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    # canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    src = to_numpy(self.env.src[i].permute(1, 2, 0))  # .repeat(3, axis=2)
                    tgt = to_numpy(self.env.tgt[i].permute(1, 2, 0))  # .repeat(3, axis=2)
                    canvas = to_numpy(self.env.canvas[i].permute(1, 2, 0))  # .repeat(3, axis=2)
                    ref = to_numpy(self.env.ref[i].permute(1, 2, 0))  # .repeat(3, axis=2)
                    self.writer.add_image(str(self.env.imgid[i]) + '/_source.png', src, log)
                    self.writer.add_image(str(self.env.imgid[i]) + '/_target.png', tgt, log)
                    self.writer.add_image(str(self.env.imgid[i]) + '/_canvas.png', canvas, log)
                    self.writer.add_image(str(self.env.imgid[i]) + '/_ref.png', ref, log)

    def step(self, action):
        with torch.no_grad():
            ob, r, d, _ = self.env.step(torch.tensor(action).to(device))
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, _

    def get_dist(self):
        return to_numpy((((self.env.tgt.float() - self.env.canvas.float()) / 255) ** 2).mean(1).mean(1).mean(1))

    def reset(self, test=False, episode=0):
        self.test = test
        ob = self.env.reset(self.test, episode * self.env_batch)
        return ob
