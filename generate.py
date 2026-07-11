# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# 論文で提案されたEDM sampler（Algorithm 2）。

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDMの標準samplerで画像を生成する。

    ノイズレベルsigmaを大きい値から0へ下げながら、各区間でまずEuler stepを
    行い、最後の区間以外ではHeun型の2次補正を加える。`S_churn` が0より
    大きい場合は、各stepの前に一時的にノイズを増やす確率的な操作も入る。
    """
    # networkが対応している範囲に合わせてノイズレベルを丸める。
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # EDMのrho scheduleでsigma列を作る。数式では
    #   t_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho)-sigma_max^(1/rho)))^rho
    # で、i=0が最大ノイズ、i=N-1が最小ノイズ。末尾にノイズなし画像を表す
    # t_N=0を追加する。
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # latent zは標準正規分布なので、初期状態は x_0 = t_0 * z とする。
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # 必要なら一時的にノイズを増やす。数式では
        #   gamma = min(S_churn/N, sqrt(2)-1)
        #   t_hat = t_cur * (1 + gamma)
        #   x_hat = x_cur + sqrt(t_hat^2 - t_cur^2) * S_noise * eps
        # で、eps ~ N(0,I)。S_churn=0ならgamma=0となり、決定的samplerとして動く。
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # denoised=D(x_hat,t_hat)からODEの向きd=(x-D)/sigmaを求め、
        # まずEuler stepで次のsigmaまで進める。対応する数式は
        #   d_cur = (x_hat - D(x_hat, t_hat)) / t_hat
        #   x_next = x_hat + (t_next - t_hat) * d_cur
        # で、これは dx/dt = (x - D(x,t)) / t を1次近似で積分している。
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 最後以外の区間では、Eulerで進めた先でもう一度向きを評価し、
        # 始点と終点の平均方向を使うHeun型の2次補正を行う。数式では
        #   d_prime = (x_next - D(x_next, t_next)) / t_next
        #   x_next = x_hat + (t_next - t_hat) * (d_cur + d_prime) / 2
        # となり、Eulerの予測方向を終点側の方向で補正する。
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# EDMのprobability flow ODEに合わせて移植したPNDM風sampler。

def pndm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM座標系でPNDM/F-PNDM風のODE solverを使ってサンプリングする。

    元のPNDM論文はDDPM/DDIMの変数で更新式を書いている。一方、EDMでは
    probability flow ODEをより直接的に扱える。ノイズレベル sigma=t において、
    denoiser D(x,t) から次のODE微分が得られる。

        dx/dt = (x - D(x,t)) / t.

    この実装ではEDMのsigma scheduleと任意のchurn stepはそのまま使い、
    EDMのHeun補正をPNDMの高次積分パターンに置き換える。まずRunge-Kuttaで
    微分履歴を作り、その後Adams-Bashforthで直近4つの微分を再利用する。
    """
    # networkが対応している範囲に合わせてノイズレベルを丸める。
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # sampler同士の比較でノイズレベルを揃えるため、edm_samplerと同じEDM rho
    # scheduleを使う。数式は上のEDM samplerと同じ
    #   t_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho)-sigma_max^(1/rho)))^rho
    # で、末尾に追加するt_N=0はノイズなし画像の終点だが、network自体は
    # sigma=0では評価しない。
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    def ode_derivative(x, t):
        # EDM denoiserの予測をprobability flow ODEの向きに変換する。
        # 数式では f(x,t) = dx/dt = (x - D(x,t)) / t。
        # PNDMではこの f(x,t) をstep間で保存して再利用する。
        denoised = net(x, t, class_labels).to(torch.float64)
        return (x - denoised) / t

    # 最初の、つまり最大のsigmaでスケールしたGaussian noiseから始める。
    # 数式では x_0 = t_0 * z, z ~ N(0,I)。
    # 以後はscheduleを逆向きに積分するため、h=t_next-t_hatは負になる。
    x_next = latents.to(torch.float64) * t_steps[0]

    # 直近のODE微分を古い順に保持する。4次Adams-Bashforthには4つ必要なので、
    # それまではRK4で安定した初期stepを作る。
    d_history = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # 決定的なODE stepの前に、EDMの任意の確率的なchurn操作を合わせる。
        # x_hatの式はEDM samplerと同じ:
        #   x_hat = x_cur + sqrt(t_hat^2 - t_cur^2) * S_noise * eps
        # S_churn=0なら t_hat=t_cur, x_hat=x_cur のままになる。
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # 現在の区間の始点で微分 f_i=f(x_hat,t_hat) を評価し、多段法に必要な
        # 直近4つだけを保持する。刻み幅は h=t_next-t_hat。
        h = t_next - t_hat
        d_cur = ode_derivative(x_hat, t_hat)
        d_history.append(d_cur)
        if len(d_history) > 4:
            d_history.pop(0)

        if i == num_steps - 1:
            # 最後の区間では終点t_nextが0になる。net(..., sigma=0)を呼ぶ
            # RK/Heun型の補正は避け、使える履歴で可能な最も高次の
            # Adams-Bashforth stepを使う。例えばAB4は
            #   x_{n+1} = x_n + h/24 * (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})
            # で、履歴が足りない場合はAB3、AB2、Eulerへ落とす。
            if len(d_history) >= 4:
                d = (55 * d_history[-1] - 59 * d_history[-2] + 37 * d_history[-3] - 9 * d_history[-4]) / 24
            elif len(d_history) == 3:
                d = (23 * d_history[-1] - 16 * d_history[-2] + 5 * d_history[-3]) / 12
            elif len(d_history) == 2:
                d = (3 * d_history[-1] - d_history[-2]) / 2
            else:
                d = d_history[-1]
            x_next = x_hat + h * d
        elif len(d_history) < 4:
            # ウォームアップ段階。PNDMではRunge-Kutta型の開始stepを使い、
            # 低次のEuler推定をいくつも履歴に入れるのではなく、
            # 後続の多段solverが信頼しやすい微分履歴から始められるようにする。
            # 4次Runge-Kuttaの式は
            #   k1=f(x,t)
            #   k2=f(x+h*k1/2, t+h/2)
            #   k3=f(x+h*k2/2, t+h/2)
            #   k4=f(x+h*k3,   t+h)
            #   x_next = x + h*(k1+2*k2+2*k3+k4)/6
            # に対応する。
            t_mid = t_hat + h * 0.5
            d_1 = d_cur
            d_2 = ode_derivative(x_hat + h * 0.5 * d_1, t_mid)
            d_3 = ode_derivative(x_hat + h * 0.5 * d_2, t_mid)
            d_4 = ode_derivative(x_hat + h * d_3, t_next)
            x_next = x_hat + h * (d_1 + 2 * d_2 + 2 * d_3 + d_4) / 6
        else:
            # PNDM/F-PNDMの主処理段階。4次Adams-Bashforthは現在の微分と
            # 過去3つの微分から次のsampleを予測する:
            #   d_AB4 = (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3}) / 24
            #   x_{n+1} = x_n + h * d_AB4
            # 毎回新しくRK/Heun補正を行う場合よりnetwork評価を節約できる。
            d = (55 * d_history[-1] - 59 * d_history[-2] + 37 * d_history[-3] - 9 * d_history[-4]) / 24
            x_next = x_hat + h * d

    return x_next

#----------------------------------------------------------------------------
# 論文Section 3のablation実験を再現するための汎用sampler。
# solver・離散化・sigma(t)・scaling s(t)を個別に切り替えられる。

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM論文の設計空間を切り替えながらサンプリングする。

    `edm_sampler` は提案手法の固定構成だが、この関数は過去手法との比較や
    論文中のablationを行うため、ODE solver、時間離散化、ノイズschedule、
    signal scalingを独立に指定できるようにしたもの。
    """
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # VP/VE系のsigma(t)、その微分、逆関数を定義する。
    # VPでは
    #   sigma(t) = sqrt(exp(0.5 * beta_d * t^2 + beta_min * t) - 1)
    # VEでは
    #   sigma(t) = sqrt(t)
    # と置く。ablationでは「どのsigma列を使うか」と「ODEをどのtで書くか」を
    # 分けて比較するため、sigmaとsigma_invの両方が必要になる。
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # 離散化方式ごとに、論文で使う標準的なsigma範囲を選ぶ。
    # 明示的にsigma_min/maxが渡された場合はそちらを優先する。
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # networkが対応している範囲に合わせてノイズレベルを丸める。
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # 選ばれたsigma_min/maxを通るようにVP scheduleのbetaを再計算する。
    # sigma(epsilon_s)=sigma_min, sigma(1)=sigma_max を満たすbeta_d, beta_minを
    # 逆算している。
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # まず「どのsigma値を通るか」を決める。ここがtime step discretizationの
    # ablation対象で、VP/VE/IDDPM/EDMそれぞれの刻み方を比較できる。
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        # VP SDE由来のtを等間隔に取り、それをsigmaへ写す。
        #   t_i = 1 + i/(N-1) * (epsilon_s - 1)
        #   sigma_i = sigma_VP(t_i)
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        # VEではsigma^2側を幾何級数的に刻む。
        #   t_i = sigma_max^2 * (sigma_min^2/sigma_max^2)^(i/(N-1))
        #   sigma_i = sqrt(t_i)
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        # iDDPMの離散ノイズ列u_jを作り、指定範囲内からnum_steps個を抜き出す。
        # コード中の再帰は、alpha_bar(j)に基づくiDDPMの離散scheduleを
        # sigma表現へ変換している。
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        # EDMのrho schedule。少ないstepでも低sigma側を細かく扱える。
        #   sigma_i = (sigma_max^(1/rho) + i/(N-1)*(sigma_min^(1/rho)-sigma_max^(1/rho)))^rho
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # ODE上の時刻tとノイズレベルsigma(t)の対応を選ぶ。
    # discretizationで決めたsigma列を、ここで選んだscheduleのtへ戻す。
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        # EDM提案設定ではsigma(t)=tとして扱う。したがって
        #   sigma'(t)=1, sigma^{-1}(sigma)=sigma
        # になる。
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # signal scaling s(t)を選ぶ。VP scalingでは状態x自体もs(t)でスケールされ、
    # noneではEDMと同じくs(t)=1として扱う。
    if scaling == 'vp':
        # VP scalingの式:
        #   s(t) = 1 / sqrt(1 + sigma(t)^2)
        #   s'(t) = -sigma(t) * sigma'(t) * s(t)^3
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # sigma列を、選択したschedule上のt列へ変換する。最後にt_N=0を足して
    # ノイズなし画像の終点まで進める。
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # 初期状態は最大ノイズsigma(t_0)にsignal scaling s(t_0)を掛けたもの。
    # 数式では x_0 = sigma(t_0) * s(t_0) * z, z ~ N(0,I)。
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # EDM samplerと同じく、指定範囲内のsigmaで一時的にノイズを増やせる。
        # scalingありの一般形では
        #   x_hat = s(t_hat)/s(t_cur) * x_cur
        #         + sqrt(sigma(t_hat)^2 - sigma(t_cur)^2) * s(t_hat) * S_noise * eps
        # となる。s(t)=1ならEDM samplerの式に戻る。
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # 選択したsigma(t)とs(t)に対応するODE右辺を評価し、Eulerで予測点へ進める。
        # 一般化された状態xのODE右辺は
        #   d = (sigma'/sigma + s'/s) * x
        #       - sigma' * s / sigma * D(x/s, sigma)
        # で、Euler予測は
        #   x_prime = x_hat + alpha * h * d_cur
        #   t_prime = t_hat + alpha * h
        # となる。
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Euler指定または最終区間ではそのまま進める。Heun指定では予測点でも
        # 右辺を評価し、2つの方向を重み付き平均して2次補正を行う。
        # alphaを含むHeun補正式は
        #   x_next = x_hat + h * ((1 - 1/(2*alpha)) * d_cur
        #                         + 1/(2*alpha) * d_prime)
        # で、alpha=1なら通常の平均 (d_cur+d_prime)/2 になる。
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# minibatch内の各sampleに別々のseedを割り当てるためのtorch.Generatorラッパー。

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        # seedごとに独立したGeneratorを作ることで、batch sizeやGPU数が変わっても
        # 同じseedの画像が再現しやすくなる。
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        # batch次元の各要素を、それぞれ対応するGeneratorから生成して積み上げる。
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        # torch.randn_likeと同じ形で使えるようにした薄いラッパー。
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        # class labelのランダム選択でもsampleごとのseedを保つ。
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# カンマ区切りのseed指定を整数listへ変換する。
# 例: '1,2,5-10' -> [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    # Clickが既にlistとして渡してきた場合は、そのまま返す。
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            # '5-10' のような範囲指定は両端を含めて展開する。
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--sampler',                 help='Sampling method', metavar='edm|pndm',                              type=click.Choice(['edm', 'pndm']), default='edm', show_default=True)
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # 64枚の画像を生成して out/*.png として保存する
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # 2 GPUで1024枚の画像を生成する
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    # torchrun実行時のrank/world sizeを初期化する。単一GPU実行でも
    # world size=1として同じコードパスで動く。
    dist.init()

    # seed列を各rankへ均等に割り当てる。rankごとに担当batchだけを処理するので、
    # 複数GPUでも出力seedの重複を避けられる。
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # rank 0が先にnetworkを取得する。URLからのdownloadやcache作成で複数rankが
    # 同時に同じファイルへ触らないようにするため。
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # pickle内にはema networkが保存されている。EDMでは通常このemaをsamplingに使う。
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # rank 0の読み込みが終わったら、他rankも同じnetworkを読み込む。
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Clickの未指定optionはNoneで入るため、samplerへ渡す前に取り除く。
    # samplerは全batchで同じなので、loop内でpopして消してしまわないように
    # ここで一度だけ解決する。関数として直接呼ばれてsampler keyが無い場合も、
    # 既定のEDM samplerへ倒す。
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    sampler = sampler_kwargs.pop('sampler', 'edm')

    # ablation samplerは独自のsolver/discretization/schedule空間を持つ。
    # PNDMはEDM座標系のsamplerとして分けておき、実験ではsolver変更だけを
    # 1つずつ比較できるようにする。
    have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
    if sampler == 'pndm' and have_ablation_kwargs:
        raise click.ClickException('--sampler=pndm cannot be combined with ablation options --solver/--disc/--schedule/--scaling')
    sampler_fn = pndm_sampler if sampler == 'pndm' else ablation_sampler if have_ablation_kwargs else edm_sampler

    # 各rankが担当するbatchだけを生成する。progress barはrank 0だけに出す。
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        # 全rankでbatch境界を揃え、分散実行時の進行を同期する。
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # seedごとにlatentを作る。StackedRandomGeneratorにより、batch分割が変わっても
        # 同じseedなら同じlatent/class labelになる。
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            # class-conditional modelでは、seedごとにランダムclassをone-hotで選ぶ。
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            # --class指定がある場合は、batch内の全sampleをそのclassへ固定する。
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # network出力は概ね[-1,1]なので、PNG保存用に[0,255]のuint8へ変換する。
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            # --subdirs指定時は、000000/001000/... のようにseed範囲ごとに分ける。
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                # grayscale modelの場合は'L'、RGB modelの場合は'RGB'で保存する。
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # 全rankの保存が終わるまで待ってから終了する。
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
