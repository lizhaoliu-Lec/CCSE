import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from DRL.rpm import rpm
from DRL.actor import *
from utils.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import cv2 as cv
import math
import pdb

N = 5

color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 0),
         (128, 0, 255), (255, 128, 0), (255, 0, 128)]
width = 128

coord = torch.zeros([1, 2, width, width]).to(device)
coord[0, 0, :, :] = torch.linspace(-1, 1, 128)
coord[0, 1, :, :] = torch.linspace(-1, 1, 128)[..., None]

criterion = nn.MSELoss()

channels = 3


class TPS(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y, point, device):
        '''
        X target 
        Y source
        '''

        #        X *= 10
        #        Y *= 10
        #        point *= 10

        n, k = X.shape[:2]

        Z = torch.zeros(n, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        W, A = Q[:, :k], Q[:, k:]
        NP = torch.ones(n, 10, 3, device=device)
        NP[:, :, 1:] = point
        D2 = torch.pow(point[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)
        new_point = (torch.matmul(NP, A) + torch.matmul(U, W))  # / 10

        return new_point


tps = TPS().to(device)


def decode(point, action, point_color=True, ref_point=None):
    action = action.view(-1, 25, 2)
    B = action.shape[0]
    source = torch.zeros(B, N, N, 2).to(device)
    source[:, :, :, 0] = torch.linspace(-2, 2, N)
    source[:, :, :, 1] = torch.linspace(-2, 2, N)[..., None]
    source = source.view(-1, N * N, 2)
    eps = 0.99
    target = source + ((action - 0.5) * eps)

    new_point = tps(source, target, point, device)

    maxx = new_point[:, :, :1].max(1)[0]
    maxy = new_point[:, :, 1:].max(1)[0]
    minx = new_point[:, :, :1].min(1)[0]
    miny = new_point[:, :, 1:].min(1)[0]
    mxy = torch.cat(((maxx + minx) / 2, (maxy + miny) / 2), 1)
    new_point = new_point - mxy[:, None, :]

    dxy = new_point.max(1)[0].max(1)[0] + 1e-4
    if point_color:
        new_point = new_point / dxy[:, None, None] * 0.5
    else:
        new_point = new_point / dxy[:, None, None]
        if not ref_point is None:
            maxx = ref_point[:, :, :1].max(1)[0]
            maxy = ref_point[:, :, 1:].max(1)[0]
            minx = ref_point[:, :, :1].min(1)[0]
            miny = ref_point[:, :, 1:].min(1)[0]
            mxy = torch.cat(((maxx + minx) / 2, (maxy + miny) / 2), 1)
            dxy = (ref_point - mxy[:, None, :]).max(1)[0].max(1)[0]
            new_point = new_point * dxy[:, None, None] + mxy[:, None, :]

    # draw
    canvas = np.zeros((B, 128, 128, 3))
    for b in range(B):
        p = new_point[b].detach().cpu().numpy()
        p = (p + 1) * 64
        p = p.astype(np.int)

        # detect nan here
        if torch.any(new_point[b].isnan()):
            print("ERROR: NAN occur in new_point[b], continue...")
            continue

        for i in range(9):
            try:
                cv.line(canvas[b], (p[i][0], p[i][1]), (p[i + 1][0], p[i + 1][1]), (255, 255, 255), 4)
            except:
                print("EXCEPT: NAN occur in new_point[b], continue...")
                print(p, new_point[b], action[b], point[b])
                continue
                # sys.exit(0)
        if point_color:
            for i in range(10):
                cv.circle(canvas[b], (p[i][0], p[i][1]), 2, color[i], -1)
    canvas = np.transpose(canvas, (0, 3, 1, 2))
    canvas = torch.tensor(canvas).to(device).float()

    return canvas, new_point


def move(point, action, W=128):
    B = action.shape[0]
    dxy = action[:, 0]
    mxy = 2 * action[:, 1:] - 1

    new_point = point * dxy[:, None, None] + mxy[:, None, :]

    canvas = np.zeros((B, W, W, 3))
    if W == 128:
        line_width = 4
    if W == 320:
        line_width = 8
    for b in range(B):
        p = new_point[b].detach().cpu().numpy()
        p = (p + 1) * (W // 2)
        p = p.astype(np.int)
        for i in range(9):
            cv.line(canvas[b], (p[i][0], p[i][1]), (p[i + 1][0], p[i + 1][1]), (255, 255, 255), line_width)
    canvas = np.transpose(canvas, (0, 3, 1, 2))
    canvas = torch.tensor(canvas).to(device).float()

    return canvas, new_point


class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size

        self.actor = ResNet(2 * channels + 3, 18, 50)  # canvas, ref, T, coord
        self.actor_target = ResNet(2 * channels + 3, 18, 50)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)

        if (resume != None):
            self.load_weights(resume)
            print(resume)

        hard_update(self.actor_target, self.actor)

        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0

        self.state = [None] * self.env_batch  # Most recent state
        self.action = [None] * self.env_batch  # Most recent action
        self.choose_device()

    def play(self, state, target=False):
        # actor
        # canvas, src, ref, T, corrd
        canvas = state[:, 0: channels].float() / 255
        src = state[:, 1 * channels: 2 * channels].float() / 255
        T = state[:, 2 * channels: 2 * channels + 1].float()
        tgt = state[:, 2 * channels + 1: 3 * channels + 1].float() / 255
        ref = state[:, 3 * channels + 1: 4 * channels + 1].float() / 255
        state2 = torch.cat((canvas, ref, T, coord.expand(state.shape[0], 2, width, width)), 1)
        if target:
            return self.actor_target(state2)
        else:
            return self.actor(state2)

    def evaluate(self, state, action, target=False):
        # state
        # canvas, src, T, tgt, ref
        canvas0 = state[:, 0: channels].float() / 255
        src = state[:, 1 * channels: 2 * channels].float() / 255
        T = state[:, 2 * channels: 2 * channels + 1].float()
        tgt = state[:, 2 * channels + 1: 3 * channels + 1].float() / 255
        ref = state[:, 3 * channels + 1: 4 * channels + 1].float() / 255
        point = state[:, 4 * channels + 1, 0:10, 0:2].float()
        tgt_point = state[:, 4 * channels + 2, 0:10, 0:2].float()
        last_point = point.clone()
        canvas1, new_point = decode(point, action)

        coord_ = coord.expand(state.shape[0], 2, width, width)

        L2_reward = ((last_point - tgt_point) ** 2).sum(2).sqrt().mean(1) - ((new_point - tgt_point) ** 2).sum(
            2).sqrt().mean(1)
        L2_reward = L2_reward.view(-1, 1)

        if self.log % 20 == 0:
            self.writer.add_scalar('train/l2_reward', L2_reward.mean(), self.log)
        return L2_reward

    def update_policy(self, lr):
        self.log += 1

        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        # Sample batch
        state, action, reward, next_state, terminal = self.memory.sample_batch(self.batch_size, device)

        action = self.play(state)
        pre_q = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()

        # train actor
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_optim.step()
        # print(policy_loss.item())

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)

        return -policy_loss

    def observe(self, reward, state, done, step):
        s0 = self.state.clone().detach().cpu()
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = state.clone().detach().cpu()
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(-noise_factor, noise_factor, action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)

        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))

    def save_model(self, path, iter=None):
        self.actor.cpu()
        if iter is None:
            save_path = '{}/actor.pkl'.format(path)
        else:
            save_path = '{}/actor-{}.pkl'.format(path, iter)
        torch.save(self.actor.state_dict(), save_path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()

    def choose_device(self):
        self.actor.to(device)
        self.actor_target.to(device)
