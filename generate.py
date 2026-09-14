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
import json
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
# Restart Sampling for Improving Generative Processes (Algorithm 2).

def parse_restart_info(restart_info):
    """Parse the public ``[N_restart, K, t_min, t_max]`` JSON format."""
    if restart_info in [None, '']:
        return []
    if isinstance(restart_info, str):
        try:
            restart_info = json.loads(restart_info)
        except json.JSONDecodeError as err:
            raise ValueError(f'invalid JSON: {err.msg}') from err
    if not isinstance(restart_info, dict):
        raise ValueError('restart_info must be a JSON object')

    levels = []
    for name, values in restart_info.items():
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(f'Restart level {name!r} must be [N_restart, K, t_min, t_max]')
        restart_steps, iterations, t_min, t_max = values
        try:
            parsed_restart_steps = int(restart_steps)
            parsed_iterations = int(iterations)
        except (TypeError, ValueError) as err:
            raise ValueError(f'Restart level {name!r} requires integer N_restart and K') from err
        if isinstance(restart_steps, bool) or parsed_restart_steps != restart_steps or parsed_restart_steps < 2:
            raise ValueError(f'Restart level {name!r} requires integer N_restart >= 2')
        if isinstance(iterations, bool) or parsed_iterations != iterations or parsed_iterations < 1:
            raise ValueError(f'Restart level {name!r} requires integer K >= 1')
        try:
            t_min = float(t_min)
            t_max = float(t_max)
        except (TypeError, ValueError) as err:
            raise ValueError(f'Restart level {name!r} requires numeric t_min and t_max') from err
        if not np.isfinite(t_min) or not np.isfinite(t_max) or not 0 < t_min < t_max:
            raise ValueError(f'Restart level {name!r} requires 0 < t_min < t_max')
        levels.append(dict(
            name=str(name),
            restart_steps=parsed_restart_steps,
            iterations=parsed_iterations,
            t_min=t_min,
            t_max=t_max,
        ))
    return levels

def restart_nfe(num_steps, restart_info):
    """Return the exact network-evaluation count for EDM Heun + Restart."""
    levels = parse_restart_info(restart_info) if not isinstance(restart_info, list) else restart_info
    return 2 * num_steps - 1 + sum(
        level['iterations'] * 2 * (level['restart_steps'] - 1)
        for level in levels
    )

def restart_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    restart_info=None, restart_noise=1,
):
    """Sample with EDM Heun and one or more finite-width Restart cycles.

    ``restart_info`` follows the authors' public format::

        {"0": [N_restart, K, t_min, t_max], ...}

    Each ``t_min`` is mapped to the nearest point in the main EDM schedule.
    The forward jump uses variance ``t_max**2 - t_min**2`` and the backward
    path is a fresh, deterministic Heun ODE solve.
    """
    if num_steps < 2:
        raise ValueError('Restart sampler requires num_steps >= 2')
    if restart_noise < 0 or not np.isfinite(restart_noise):
        raise ValueError('restart_noise must be a finite non-negative number')

    levels = parse_restart_info(restart_info) if not isinstance(restart_info, list) else restart_info
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    levels_by_index = {}
    for level in levels:
        if level['t_max'] > sigma_max:
            raise ValueError(
                f"Restart level {level['name']!r} has t_max={level['t_max']} above supported sigma_max={sigma_max}")
        restart_idx = int(torch.argmin((t_steps - level['t_min']).abs()).item())
        if restart_idx == 0 or restart_idx >= len(t_steps) - 1:
            raise ValueError(
                f"Restart level {level['name']!r} maps t_min={level['t_min']} to an invalid main schedule endpoint")
        if restart_idx in levels_by_index:
            other = levels_by_index[restart_idx]['name']
            raise ValueError(
                f"Restart levels {other!r} and {level['name']!r} map to the same main schedule index {restart_idx}")

        actual_t_min = t_steps[restart_idx]
        if level['t_max'] <= float(actual_t_min):
            raise ValueError(
                f"Restart level {level['name']!r} requires t_max above its rounded t_min={float(actual_t_min):.8g}")
        restart_indices = torch.arange(level['restart_steps'], dtype=torch.float64, device=latents.device)
        restart_steps = (
            level['t_max'] ** (1 / rho)
            + restart_indices / (level['restart_steps'] - 1)
            * (actual_t_min ** (1 / rho) - level['t_max'] ** (1 / rho))
        ) ** rho
        restart_steps = net.round_sigma(restart_steps)
        if not bool(torch.all(restart_steps[:-1] > restart_steps[1:])):
            raise ValueError(f"Restart level {level['name']!r} does not produce a strictly descending schedule")
        levels_by_index[restart_idx] = dict(level, t_steps=restart_steps)

    def heun_step(x_cur, t_cur, t_next, apply_churn=False):
        gamma = 0
        if apply_churn and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if t_next != 0:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_next = heun_step(x_next, t_cur, t_next, apply_churn=True)

        restart_idx = i + 1
        if restart_idx not in levels_by_index:
            continue
        level = levels_by_index[restart_idx]
        restart_steps = level['t_steps']
        noise_std = (restart_steps[0] ** 2 - restart_steps[-1] ** 2).sqrt() * restart_noise
        for _ in range(level['iterations']):
            x_next = x_next + noise_std * randn_like(x_next)
            for restart_cur, restart_next in zip(restart_steps[:-1], restart_steps[1:]):
                x_next = heun_step(x_next, restart_cur, restart_next)

    return x_next

#----------------------------------------------------------------------------
# EDMのprobability flow ODEを古典的4次Runge-Kuttaで解くsampler。

def rk4_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM座標系で古典的4次Runge-Kutta法を使ってサンプリングする。

    EDM samplerと同じsigma scheduleとchurn操作を使い、各区間のODE

        dx/dt = (x - D(x,t)) / t

    をRK4で積分する。最後の区間は t_next=0 なので network を sigma=0 で
    評価せず、始点の微分によるEuler stepで閉じる。
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    def ode_derivative(x, t):
        denoised = net(x, t, class_labels).to(torch.float64)
        return (x - denoised) / t

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        h = t_next - t_hat
        d_1 = ode_derivative(x_hat, t_hat)
        if i == num_steps - 1:
            x_next = x_hat + h * d_1
            continue

        t_mid = t_hat + h * 0.5
        d_2 = ode_derivative(x_hat + h * 0.5 * d_1, t_mid)
        d_3 = ode_derivative(x_hat + h * 0.5 * d_2, t_mid)
        d_4 = ode_derivative(x_hat + h * d_3, t_next)
        x_next = x_hat + h * (d_1 + 2 * d_2 + 2 * d_3 + d_4) / 6

    return x_next

#----------------------------------------------------------------------------
# EDMのprobability flow ODEに合わせて移植したPNDM風sampler。

def pndm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    linear_schedule=False, rk_warmup=True,
):
    """EDM座標系でPNDM/F-PNDM風のODE solverを使ってサンプリングする。

    元のPNDM論文はDDPM/DDIMの変数で更新式を書いている。一方、EDMでは
    probability flow ODEをより直接的に扱える。ノイズレベル sigma=t において、
    denoiser D(x,t) から次のODE微分が得られる。

        dx/dt = (x - D(x,t)) / t.

    この実装ではEDMのsigma scheduleと任意のchurn stepはそのまま使い、
    EDMのHeun補正をPNDMの高次積分パターンに置き換える。既定では
    Runge-Kuttaで微分履歴を作り、その後Adams-Bashforthで直近4つの微分を
    再利用する。rk_warmup=FalseではEuler, AB2, AB3で履歴を立ち上げる。
    """
    # networkが対応している範囲に合わせてノイズレベルを丸める。
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if linear_schedule:
        # 固定係数Adams-Bashforthの等間隔step仮定に合わせるため、
        # sigma_maxから0までをnum_steps区間で線形に刻む。networkは最後の
        # sigma=0では評価せず、直前の正のsigmaで得た微分から最終stepを行う。
        step_indices = torch.arange(num_steps + 1, dtype=torch.float64, device=latents.device)
        t_steps = sigma_max * (1 - step_indices / num_steps)
        t_steps = torch.cat([net.round_sigma(t_steps[:-1]), torch.zeros_like(t_steps[:1])]) # t_N = 0
    else:
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

        if i == num_steps - 1 or (not rk_warmup and len(d_history) < 4):
            # no-RK warmupでは履歴数に応じてEuler, AB2, AB3を使う。最後の
            # 区間でもnet(..., sigma=0)を避け、使える履歴で可能な最も高次の
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
# EDMのprobability flow ODEに合わせて移植したS-PNDM風sampler。

def spndm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM座標系でS-PNDM風の2次ODE solverを使ってサンプリングする。

    公式S-PNDMは初回をimproved Eulerで始め、その後は2次Adams-Bashforth
    でnoise予測の履歴を使う。ここではEDMのprobability flow ODE

        f(x,t) = dx/dt = (x - D(x,t)) / t

    を対象にして、初回はHeun型2次補正、2回目以降は

        f_AB2 = (3 f_n - f_{n-1}) / 2
        x_{n+1} = x_n + h * f_AB2

    で更新する。
    """
    # networkが対応している範囲に合わせてノイズレベルを丸める。
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # PNDM/F-PNDM版と同じく、EDM samplerと同じrho scheduleを使う。
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    def ode_derivative(x, t):
        # EDM denoiserの予測からODE右辺 f(x,t)=(x-D(x,t))/t を作る。
        denoised = net(x, t, class_labels).to(torch.float64)
        return (x - denoised) / t

    # x_0 = t_0 * z から開始し、sigmaを下げる方向へ積分する。
    x_next = latents.to(torch.float64) * t_steps[0]
    d_prev = None
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # EDMと同じchurn操作。S_churn=0なら決定的なS-PNDMになる。
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        h = t_next - t_hat
        d_cur = ode_derivative(x_hat, t_hat)

        if d_prev is None:
            # 初回は履歴が無いので、公式S-PNDMのimproved Eulerに対応する
            # Heun型補正で開始する。最終区間だけはsigma=0評価を避けてEulerにする。
            x_next = x_hat + h * d_cur
            if i < num_steps - 1:
                d_prime = ode_derivative(x_next, t_next)
                x_next = x_hat + h * (0.5 * d_cur + 0.5 * d_prime)
        else:
            # 2回目以降は2次Adams-Bashforth:
            #   x_{n+1} = x_n + h * (3 f_n - f_{n-1}) / 2
            d = (3 * d_cur - d_prev) / 2
            x_next = x_hat + h * d

        d_prev = d_cur

    return x_next

#----------------------------------------------------------------------------
# EDMのprobability flow ODEをAdams-Bashforth多段法で解くsampler。

def ab_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM座標系で明示的Adams-Bashforth多段法を使ってサンプリングする。

    対象のODEは他のEDM座標系samplerと同じ

        dx/dt = (x - D(x,t)) / t.

    純粋なmultiple-step法として、各stepでは始点のnetwork評価を1回だけ行い、
    直近の微分履歴を使って更新する。履歴が少ない序盤はEuler, AB2, AB3と
    低次から始め、4本そろった後はAB4を使う。EDMのrho scheduleは非等間隔
    なので、固定係数ではなく各区間のLagrange補間を積分して係数を作る。
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    def ode_derivative(x, t):
        denoised = net(x, t, class_labels).to(torch.float64)
        return (x - denoised) / t

    def ab_coefficients(t_start, t_end, times):
        # Lagrange基底 L_j(t) を [t_start, t_end] で積分し、
        # x_{n+1}=x_n+sum_j coeff_j*f(t_j,x_j) の係数にする。
        coeffs = []
        one = torch.ones((), dtype=torch.float64, device=latents.device)
        zero = torch.zeros((), dtype=torch.float64, device=latents.device)
        for j, t_j in enumerate(times):
            poly = [one]
            for m, t_m in enumerate(times):
                if m == j:
                    continue
                denom = t_j - t_m
                updated = [zero for _ in range(len(poly) + 1)]
                for degree, value in enumerate(poly):
                    updated[degree] = updated[degree] - value * t_m / denom
                    updated[degree + 1] = updated[degree + 1] + value / denom
                poly = updated
            integral = zero
            for degree, value in enumerate(poly):
                integral = integral + value * (t_end ** (degree + 1) - t_start ** (degree + 1)) / (degree + 1)
            coeffs.append(integral)
        return coeffs

    x_next = latents.to(torch.float64) * t_steps[0]
    t_history = []
    d_history = []
    max_order = 4
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        t_history.append(t_hat)
        d_history.append(ode_derivative(x_hat, t_hat))
        if len(t_history) > max_order:
            t_history.pop(0)
            d_history.pop(0)

        coeffs = ab_coefficients(t_hat, t_next, t_history)
        x_next = x_hat
        for coeff, d_cur in zip(coeffs, d_history):
            x_next = x_next + coeff * d_cur

    return x_next

#----------------------------------------------------------------------------
# Linear-path flow matching (FM) 用のsampler。
#
# 学習側 (training.loss.FMLoss / training.networks.FMPrecond) と同じ規約で、
# t=0がGaussian noise、t=1がデータになる向きの線形経路
#
#     x_t = (1 - t) z + t y
#
# を仮定する。networkはこの経路の速度場 v(x_t, t) を直接出力するので、
# 解くべきODEは
#
#     dx/dt = v(x, t),   x(0) = z ~ N(0,I)
#
# となり、EDMのようなsigma scheduleやchurn操作は使わない。時間刻みは
# t=0から1までのuniformなN区間で、`--steps=N` がその区間数を指す。

def fm_velocity(net, x, t, class_labels):
    # networkはbatchごとの時刻を受け取るので、スカラーtをbatch分に広げる。
    t_batch = t.to(torch.float64).repeat(x.shape[0])
    return net(x, t_batch, class_labels).to(torch.float64)

def fm_time_steps(num_steps, device):
    # t=0 (noise) から t=1 (data) までをnum_steps区間で等分する。
    return torch.linspace(0, 1, num_steps + 1, dtype=torch.float64, device=device)

def fm_euler_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18,
):
    """FM velocity場をEuler法で t=0 -> 1 へ積分する。

    各区間で始点の速度場を1回だけ評価するので、network評価回数 (NFE) は
    `num_steps` と一致する。`randn_like` は決定的ODEでは使わないが、
    generate.py側の共通呼び出しに合わせて受け取る。
    """
    t_steps = fm_time_steps(num_steps, latents.device)

    # FMではlatentがそのまま t=0 の状態になる。EDM samplerのような
    # sigma_maxによるスケーリングは行わない。
    x_next = latents.to(torch.float64)
    for i in range(num_steps): # 0, ..., N-1
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        x_cur = x_next

        # Euler step: x_{n+1} = x_n + (t_{n+1} - t_n) * v(x_n, t_n)
        d_cur = fm_velocity(net, x_cur, t_cur, class_labels)
        x_next = x_cur + (t_next - t_cur) * d_cur

    return x_next

def fm_heun_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18,
):
    """FM velocity場をHeun法で t=0 -> 1 へ積分する。

    EDM samplerと同じく最後の区間だけEuler stepで閉じるので、network評価
    回数 (NFE) は `2 * num_steps - 1` になる。
    """
    t_steps = fm_time_steps(num_steps, latents.device)

    x_next = latents.to(torch.float64)
    for i in range(num_steps): # 0, ..., N-1
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        x_cur = x_next
        h = t_next - t_cur

        # まずEulerで予測点へ進める。
        #   d_cur  = v(x_n, t_n)
        #   x_pred = x_n + h * d_cur
        d_cur = fm_velocity(net, x_cur, t_cur, class_labels)
        x_next = x_cur + h * d_cur

        # 最後以外の区間では予測点でも速度場を評価し、両端の平均で補正する。
        #   d_prime = v(x_pred, t_{n+1})
        #   x_{n+1} = x_n + h * (d_cur + d_prime) / 2
        if i < num_steps - 1:
            d_prime = fm_velocity(net, x_next, t_next, class_labels)
            x_next = x_cur + h * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# samplerとcheckpointの組合せ検証。
#
# FM samplerはvelocity checkpoint (training.networks.FMPrecond) 専用で、
# EDM系samplerはdenoiser checkpoint専用。取り違えると無言で壊れた画像が
# 出るだけなので、生成前に明示的なエラーにする。

FM_SAMPLERS = ['fm-euler', 'fm-heun']
RESTART_OPTIONS = ['restart_info', 'restart_noise']

# FM samplerが受け付けないEDM固有option。値が既定から変わっていれば、
# 効かないoptionを黙って無視せずエラーにする。
FM_UNSUPPORTED_OPTIONS = [
    ('sigma_min', None),
    ('sigma_max', None),
    ('rho', 7),
    ('S_churn', 0),
    ('S_min', 0),
    ('S_max', float('inf')),
    ('S_noise', 1),
]

ABLATION_OPTIONS = ['solver', 'discretization', 'schedule', 'scaling']

def is_velocity_network(net):
    # FMPrecondはprediction_type='velocity'をpickleへ保存する。EDM系の
    # checkpointにはこの属性が無いのでdenoiser扱いになる。
    return getattr(net, 'prediction_type', 'denoiser') == 'velocity'

def check_sampler_compatibility(sampler, net, sampler_kwargs):
    velocity_net = is_velocity_network(net)
    used_ablation = [f'--{key}' for key in ABLATION_OPTIONS if key in sampler_kwargs]
    used_restart = [f'--{key}' for key in RESTART_OPTIONS if key in sampler_kwargs]

    if sampler not in FM_SAMPLERS:
        if velocity_net:
            raise click.ClickException(
                f'--sampler={sampler} requires an EDM denoiser checkpoint, but the given network '
                f'predicts velocity; use --sampler=fm-euler or --sampler=fm-heun')
        if sampler == 'restart' and used_ablation:
            raise click.ClickException(
                '--sampler=restart cannot be combined with ablation options ' + '/'.join(used_ablation))
        if sampler != 'restart' and used_restart:
            raise click.ClickException(
                f'--sampler={sampler} does not use Restart options; remove ' + ' '.join(used_restart))
        return

    if not velocity_net:
        raise click.ClickException(
            f'--sampler={sampler} requires a flow matching checkpoint trained with --precond=fm, '
            f'but the given network is an EDM denoiser')
    if used_restart:
        raise click.ClickException(
            f'--sampler={sampler} does not use Restart options; remove ' + ' '.join(used_restart))
    if used_ablation:
        raise click.ClickException(
            f'--sampler={sampler} cannot be combined with ablation options ' + '/'.join(used_ablation))

    unsupported = [f'--{key}={sampler_kwargs[key]}' for key, default in FM_UNSUPPORTED_OPTIONS
                   if key in sampler_kwargs and sampler_kwargs[key] != default]
    if unsupported:
        raise click.ClickException(
            f'--sampler={sampler} integrates t from 0 to 1 and does not use EDM noise levels; '
            f'remove ' + ' '.join(unsupported))

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

@click.option('--restart_info',            help='Restart JSON: {"0":[N_restart,K,t_min,t_max]}', metavar='JSON')
@click.option('--restart_noise',           help='Restart forward-noise multiplier', metavar='FLOAT',                type=click.FloatRange(min=0))

@click.option('--sampler',                 help='Sampling method', metavar='edm|restart|rk4|pndm|pndm-no-rk|pndm-linear|spndm|ab|fm-euler|fm-heun', type=click.Choice(['edm', 'restart', 'rk4', 'pndm', 'pndm-no-rk', 'pndm-linear', 'spndm', 's-pndm', 'ab', 'adams-bashforth', 'fm-euler', 'fm-heun']), default='edm', show_default=True)
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

    \b
    # flow matching checkpoint (--precond=fm) からFM Eulerで生成する
    python generate.py --outdir=out --seeds=0-63 --batch=64 --sampler=fm-euler --steps=18 \\
        --network=training-runs/fm-linear/network-snapshot-000050.pkl
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
    if sampler == 's-pndm':
        sampler = 'spndm'
    if sampler == 'adams-bashforth':
        sampler = 'ab'
    if sampler == 'pndm-no-rk':
        sampler = 'pndm'
        sampler_kwargs['rk_warmup'] = False
    if sampler == 'pndm-linear':
        sampler = 'pndm'
        sampler_kwargs['linear_schedule'] = True

    # FM samplerはvelocity checkpoint専用、EDM系samplerはdenoiser checkpoint専用。
    # 取り違えやEDM固有optionの指定はここで明示的なエラーにする。
    check_sampler_compatibility(sampler, net, sampler_kwargs)
    if sampler in FM_SAMPLERS:
        # FM samplerはt=0->1のuniformな区間数だけを受け取る。
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if key == 'num_steps'}
        sampler_fn = {'fm-euler': fm_euler_sampler, 'fm-heun': fm_heun_sampler}[sampler]
        dist.print0(f'Using {sampler} sampler with {sampler_kwargs.get("num_steps", 18)} uniform steps from t=0 to t=1')
    else:
        if sampler == 'restart':
            try:
                sampler_kwargs['restart_info'] = parse_restart_info(sampler_kwargs.get('restart_info'))
            except ValueError as err:
                raise click.ClickException(f'Invalid --restart_info: {err}') from err
            dist.print0(
                f'Using Restart sampler with '
                f'{restart_nfe(sampler_kwargs.get("num_steps", 18), sampler_kwargs["restart_info"])} NFE '
                f'and {len(sampler_kwargs["restart_info"])} level(s)')
        # ablation samplerは独自のsolver/discretization/schedule空間を持つ。
        # RK4/PNDM系はEDM座標系のsamplerとして分けておき、実験ではsolver変更だけを
        # 1つずつ比較できるようにする。
        have_ablation_kwargs = any(x in sampler_kwargs for x in ABLATION_OPTIONS)
        if sampler in ['rk4', 'pndm', 'spndm', 'ab'] and have_ablation_kwargs:
            raise click.ClickException('--sampler=rk4/pndm/spndm/ab cannot be combined with ablation options --solver/--disc/--schedule/--scaling')
        sampler_fn = {'restart': restart_sampler, 'rk4': rk4_sampler, 'pndm': pndm_sampler, 'spndm': spndm_sampler, 'ab': ab_sampler}.get(sampler, ablation_sampler if have_ablation_kwargs else edm_sampler)

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
