#!/usr/bin/env python
"""Experiment 7-1: linear-path flow matching (FM) の実装を検証するunit test。

GPUもデータセットも使わず、CPU上の小さなtensorとstub networkだけで
以下を確認する。

- linear pathの両端が noise (t=0) と data (t=1) に一致する。
- 定数velocity場をEuler/Heunで積分すると解析解に一致する。
- Euler/Heunのnetwork呼び出し回数がNFEの定義と一致する。
- FM/EDM checkpointとsamplerの誤った組合せを拒否する。

実行:
    python experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/test_fm_linear_baseline.py
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import click                                            # noqa: E402
import generate                                         # noqa: E402
from training.loss import FMLoss                        # noqa: E402

#----------------------------------------------------------------------------
# テスト用のstub network。generate.pyのsamplerは net(x, t, class_labels) を
# 呼ぶだけなので、torch.nn.Moduleである必要はない。

class ConstantVelocityNet:
    """常に同じ速度場を返すstub network。解析解と比較するために使う。"""

    prediction_type = 'velocity'

    def __init__(self, velocity):
        self.velocity = velocity
        self.num_calls = 0
        self.call_times = []

    def __call__(self, x, t, class_labels=None, **kwargs):
        self.num_calls += 1
        self.call_times.append(float(t.reshape(-1)[0]))
        return torch.full_like(x, self.velocity)

class LinearTimeVelocityNet:
    """v(x,t) = a + b t を返すstub network。Heunの2次精度を確認するために使う。"""

    prediction_type = 'velocity'

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.num_calls = 0

    def __call__(self, x, t, class_labels=None, **kwargs):
        self.num_calls += 1
        t = t.to(torch.float64).reshape(-1, 1, 1, 1)
        return torch.full_like(x, self.a) + self.b * t

class SingleDataOracleNet:
    """データが1点 y* だけの場合の最適velocity場を返すstub network。

    x_t = (1-t) z + t y* から z を消すと

        v(x, t) = (y* - x) / (1 - t)

    になる。学習を挟まずに、FMLossの経路定義とsamplerの時間の向き・刻みが
    整合しているかを端から端まで確認するために使う。
    """

    prediction_type = 'velocity'

    def __init__(self, target):
        self.target = target

    def __call__(self, x, t, class_labels=None, **kwargs):
        t = t.to(torch.float64).reshape(-1, 1, 1, 1)
        return (self.target.to(torch.float64) - x) / (1 - t)

class DenoiserNet:
    """EDM checkpoint相当のstub network。prediction_type属性を持たない。"""

    sigma_min = 0.002
    sigma_max = 80

    def __call__(self, x, sigma, class_labels=None, **kwargs):
        return torch.zeros_like(x)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------

class TestLinearPath(unittest.TestCase):
    """x_t = (1-t) z + t y と v = y - z の定義を確認する。"""

    def setUp(self):
        torch.manual_seed(0)
        self.loss_fn = FMLoss(path='linear')
        self.y = torch.randn([4, 3, 8, 8])
        self.z = torch.randn([4, 3, 8, 8])

    def test_path_starts_at_noise(self):
        t = torch.zeros([4, 1, 1, 1])
        x_t = self.loss_fn.sample_path(self.y, self.z, t)
        self.assertTrue(torch.allclose(x_t, self.z))

    def test_path_ends_at_data(self):
        t = torch.ones([4, 1, 1, 1])
        x_t = self.loss_fn.sample_path(self.y, self.z, t)
        self.assertTrue(torch.allclose(x_t, self.y))

    def test_velocity_target_is_data_minus_noise(self):
        t = torch.rand([4, 1, 1, 1])
        v = self.loss_fn.velocity_target(self.y, self.z, t)
        self.assertTrue(torch.allclose(v, self.y - self.z))

    def test_velocity_target_matches_finite_difference(self):
        # 数値微分 d x_t / dt が velocity_target と一致することを確認する。
        # 差分の桁落ちを避けるため float64 で評価する。
        y = self.y.to(torch.float64)
        z = self.z.to(torch.float64)
        t = torch.full([4, 1, 1, 1], 0.3, dtype=torch.float64)
        eps = 1e-6
        x_plus = self.loss_fn.sample_path(y, z, t + eps)
        x_minus = self.loss_fn.sample_path(y, z, t - eps)
        finite_difference = (x_plus - x_minus) / (2 * eps)
        v = self.loss_fn.velocity_target(y, z, t)
        self.assertTrue(torch.allclose(finite_difference, v, atol=1e-6))

    def test_loss_is_zero_for_perfect_velocity(self):
        # 正解の速度場を返すnetworkに対しては loss が 0 になる。
        class PerfectNet:
            def __init__(self, loss_fn):
                self.loss_fn = loss_fn

            def __call__(self, x_t, t, labels=None, **kwargs):
                # x_t と t から z と y を復元できる形にはなっていないので、
                # ここでは呼び出し規約 (x_t, t, labels) の確認だけを行い、
                # 直前に記録した目標速度を返す。
                return self.target

        loss_fn = FMLoss()
        net = PerfectNet(loss_fn)

        original_velocity_target = loss_fn.velocity_target

        def capture(y, z, t):
            net.target = original_velocity_target(y, z, t)
            return net.target

        loss_fn.velocity_target = capture
        loss = loss_fn(net=net, images=self.y, labels=None)
        self.assertEqual(loss.shape, self.y.shape)
        self.assertLess(float(loss.max()), 1e-12)

    def test_loss_shape_and_finiteness(self):
        # 学習ループは loss.sum() を取るので、ピクセル単位のtensorを返す。
        class ZeroNet:
            def __call__(self, x_t, t, labels=None, **kwargs):
                return torch.zeros_like(x_t)

        loss = FMLoss()(net=ZeroNet(), images=self.y, labels=None)
        self.assertEqual(loss.shape, self.y.shape)
        self.assertTrue(torch.isfinite(loss).all())

    def test_rejects_unknown_path(self):
        with self.assertRaises(ValueError):
            FMLoss(path='ot')

#----------------------------------------------------------------------------

class TestFMSamplerAccuracy(unittest.TestCase):
    """定数velocity場に対する解析解 x(1) = x(0) + v との一致を確認する。"""

    def setUp(self):
        torch.manual_seed(0)
        self.latents = torch.randn([2, 3, 8, 8])

    def test_euler_matches_analytic_solution(self):
        net = ConstantVelocityNet(velocity=0.75)
        x = generate.fm_euler_sampler(net, self.latents, num_steps=7)
        expected = self.latents.to(torch.float64) + 0.75
        self.assertTrue(torch.allclose(x, expected, atol=1e-10))

    def test_heun_matches_analytic_solution(self):
        net = ConstantVelocityNet(velocity=-0.25)
        x = generate.fm_heun_sampler(net, self.latents, num_steps=7)
        expected = self.latents.to(torch.float64) - 0.25
        self.assertTrue(torch.allclose(x, expected, atol=1e-10))

    def test_heun_error_is_second_order_for_time_linear_velocity(self):
        # v(t) = a + b t の解析解は x(1) = x(0) + a + b/2。
        # Heunは台形則と等価なので各区間で厳密だが、EDMと同じく最後の区間
        # だけEulerで閉じるため、その1区間分の誤差 b/(2N^2) だけが残る。
        # Eulerは全区間で誤差が積み上がり b/(2N) になる。
        a, b, num_steps = 0.3, 1.5, 5
        expected = self.latents.to(torch.float64) + a + b / 2

        heun = generate.fm_heun_sampler(LinearTimeVelocityNet(a, b), self.latents, num_steps=num_steps)
        heun_error = float((heun - expected).abs().max())
        self.assertAlmostEqual(heun_error, b / (2 * num_steps ** 2), places=10)

        euler = generate.fm_euler_sampler(LinearTimeVelocityNet(a, b), self.latents, num_steps=num_steps)
        euler_error = float((euler - expected).abs().max())
        self.assertAlmostEqual(euler_error, b / (2 * num_steps), places=10)

        self.assertLess(heun_error, euler_error)

    def test_samplers_start_from_raw_latents(self):
        # FMではlatentがそのまま t=0 の状態になる (EDMのようなsigma_maxスケーリングは無い)。
        net = ConstantVelocityNet(velocity=0.0)
        x = generate.fm_euler_sampler(net, self.latents, num_steps=3)
        self.assertTrue(torch.allclose(x, self.latents.to(torch.float64), atol=1e-12))

    def test_oracle_field_transports_noise_to_data(self):
        # FMLossの経路定義から導かれる最適velocity場を使うと、任意のlatentが
        # データ点 y* へ運ばれる。loss側の (1-t) z + t y という向きと、
        # sampler側の t=0 -> 1 の積分方向が一致していることの確認になる。
        target = torch.full([1, 3, 8, 8], 0.4)
        for sampler_fn in [generate.fm_euler_sampler, generate.fm_heun_sampler]:
            with self.subTest(sampler=sampler_fn.__name__):
                x = sampler_fn(SingleDataOracleNet(target), self.latents, num_steps=6)
                expected = target.to(torch.float64).expand_as(x)
                self.assertTrue(torch.allclose(x, expected, atol=1e-9))

    def test_oracle_field_matches_loss_path_at_intermediate_times(self):
        # 途中のtでも、samplerの状態が学習時の経路 x_t = (1-t) z + t y* に載る。
        target = torch.full([1, 3, 8, 8], -0.2)
        loss_fn = FMLoss(path='linear')
        num_steps = 4
        x = self.latents.to(torch.float64)
        t_steps = generate.fm_time_steps(num_steps, x.device)
        for i in range(num_steps):
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            v = SingleDataOracleNet(target)(x, t_cur)
            x = x + (t_next - t_cur) * v
            expected = loss_fn.sample_path(target.to(torch.float64), self.latents.to(torch.float64), t_next)
            self.assertTrue(torch.allclose(x, expected, atol=1e-9))

    def test_time_grid_is_uniform_from_zero_to_one(self):
        net = ConstantVelocityNet(velocity=0.0)
        generate.fm_euler_sampler(net, self.latents, num_steps=4)
        self.assertEqual([round(t, 6) for t in net.call_times], [0.0, 0.25, 0.5, 0.75])

#----------------------------------------------------------------------------

class TestFMSamplerNFE(unittest.TestCase):
    """`--steps=N` に対するnetwork評価回数がNFEの定義と一致することを確認する。"""

    def setUp(self):
        torch.manual_seed(0)
        self.latents = torch.randn([2, 3, 8, 8])

    def test_euler_uses_n_evaluations(self):
        for num_steps in [1, 5, 18, 35]:
            with self.subTest(num_steps=num_steps):
                net = ConstantVelocityNet(velocity=1.0)
                generate.fm_euler_sampler(net, self.latents, num_steps=num_steps)
                self.assertEqual(net.num_calls, num_steps)

    def test_heun_uses_2n_minus_1_evaluations(self):
        for num_steps in [1, 5, 18, 35]:
            with self.subTest(num_steps=num_steps):
                net = ConstantVelocityNet(velocity=1.0)
                generate.fm_heun_sampler(net, self.latents, num_steps=num_steps)
                self.assertEqual(net.num_calls, 2 * num_steps - 1)

#----------------------------------------------------------------------------

class TestSamplerCheckpointCompatibility(unittest.TestCase):
    """samplerとcheckpointの組合せ検証を確認する。"""

    def test_fm_sampler_rejects_denoiser_checkpoint(self):
        for sampler in generate.FM_SAMPLERS:
            with self.subTest(sampler=sampler):
                with self.assertRaises(click.ClickException):
                    generate.check_sampler_compatibility(sampler, DenoiserNet(), {'num_steps': 18})

    def test_edm_samplers_reject_velocity_checkpoint(self):
        for sampler in ['edm', 'rk4', 'pndm', 'spndm', 'ab']:
            with self.subTest(sampler=sampler):
                with self.assertRaises(click.ClickException):
                    generate.check_sampler_compatibility(sampler, ConstantVelocityNet(1.0), {'num_steps': 18})

    def test_matching_combinations_are_accepted(self):
        generate.check_sampler_compatibility('fm-euler', ConstantVelocityNet(1.0), {'num_steps': 18})
        generate.check_sampler_compatibility('fm-heun', ConstantVelocityNet(1.0), {'num_steps': 18})
        generate.check_sampler_compatibility('edm', DenoiserNet(), {'num_steps': 18})

    def test_fm_sampler_rejects_ablation_options(self):
        for key in generate.ABLATION_OPTIONS:
            with self.subTest(option=key):
                with self.assertRaises(click.ClickException):
                    generate.check_sampler_compatibility(
                        'fm-euler', ConstantVelocityNet(1.0), {'num_steps': 18, key: 'euler'})

    def test_fm_sampler_rejects_non_default_edm_options(self):
        non_default = {
            'sigma_min': 0.002,
            'sigma_max': 80,
            'rho': 5,
            'S_churn': 40,
            'S_min': 0.05,
            'S_max': 50,
            'S_noise': 1.003,
        }
        for key, value in non_default.items():
            with self.subTest(option=key):
                with self.assertRaises(click.ClickException):
                    generate.check_sampler_compatibility(
                        'fm-euler', ConstantVelocityNet(1.0), {'num_steps': 18, key: value})

    def test_fm_sampler_accepts_default_edm_options(self):
        # generate.pyは既定値をそのままsampler_kwargsへ入れて渡すので、
        # 既定値のままなら通す必要がある。
        defaults = {'num_steps': 18, 'rho': 7, 'S_churn': 0, 'S_min': 0, 'S_max': float('inf'), 'S_noise': 1}
        generate.check_sampler_compatibility('fm-euler', ConstantVelocityNet(1.0), defaults)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity=2)

#----------------------------------------------------------------------------
