# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Loss function for linear-path flow matching (FM).
#
# 独立に標本化したGaussian noise z と画像 y を線形経路
#
#     x_t = (1 - t) z + t y,   t ~ U[0,1)
#
# で結び、その経路に沿った速度場
#
#     v_target = d x_t / dt = y - z
#
# をnetworkに回帰させる。EDMのようなノイズレベル依存のpreconditioningや
# loss weightingは使わず、重み1の velocity MSE をそのまま最適化する。

@persistence.persistent_class
class FMLoss:
    def __init__(self, path='linear'):
        # 現状は独立couplingのlinear pathのみ。将来Minibatch OTなどの
        # couplingを追加するときは、この属性で経路を切り替える。
        if path != 'linear':
            raise ValueError(f"FMLoss only supports path='linear', got {path!r}")
        self.path = path

    def sample_noise(self, y):
        # 独立coupling。Minibatch OTを入れる場合はここでyとzの対応を組み替える。
        return torch.randn_like(y)

    def sample_path(self, y, z, t):
        # x_t = (1 - t) z + t y。t=0でnoise、t=1でdataになる。
        return (1 - t) * z + t * y

    def velocity_target(self, y, z, t):
        # linear pathではd x_t/dt = y - z で、tには依存しない。
        return y - z

    def __call__(self, net, images, labels=None, augment_pipe=None):
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        t = torch.rand([y.shape[0], 1, 1, 1], device=y.device)
        z = self.sample_noise(y)
        x_t = self.sample_path(y, z, t)
        v_target = self.velocity_target(y, z, t)
        v_pred = net(x_t, t, labels, augment_labels=augment_labels)
        # 既存の学習ループはloss.sum()を取るので、ピクセル単位のtensorを返す。
        loss = (v_pred - v_target) ** 2
        return loss

#----------------------------------------------------------------------------
