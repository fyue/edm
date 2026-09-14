#!/usr/bin/env python
"""CPU-only regression tests for the Experiment 8 Restart sampler."""

import os
import sys
import unittest

import torch
from click.testing import CliRunner

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import click                                            # noqa: E402
import generate                                         # noqa: E402


class CountingDenoiser:
    sigma_min = 0.002
    sigma_max = 80

    def __init__(self, return_input=False):
        self.num_calls = 0
        self.return_input = return_input

    def __call__(self, x, sigma, class_labels=None, **kwargs):
        self.num_calls += 1
        return x if self.return_input else torch.zeros_like(x)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class VelocityNet:
    prediction_type = 'velocity'


def seeded_randn_like(seed):
    generator = torch.Generator().manual_seed(seed)

    def randn_like(x):
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    return randn_like


class TestRestartConfig(unittest.TestCase):
    def test_empty_config_is_baseline(self):
        self.assertEqual(generate.parse_restart_info(None), [])
        self.assertEqual(generate.parse_restart_info('{}'), [])

    def test_public_json_format(self):
        levels = generate.parse_restart_info('{"0":[3,2,0.06,0.30]}')
        self.assertEqual(levels, [dict(
            name='0', restart_steps=3, iterations=2, t_min=0.06, t_max=0.30)])
        self.assertEqual(generate.restart_nfe(18, levels), 43)

    def test_rejects_invalid_configs(self):
        invalid = [
            '[]',
            '{"0":[1,2,0.06,0.30]}',
            '{"0":[3,0,0.06,0.30]}',
            '{"0":[3,2,0.30,0.06]}',
            '{"0":[3,2,0.06]}',
            '{"0":[null,2,0.06,0.30]}',
            'not-json',
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                generate.parse_restart_info(value)


class TestRestartSampler(unittest.TestCase):
    def setUp(self):
        self.latents = torch.randn([2, 3, 4, 4], generator=torch.Generator().manual_seed(11))

    def test_no_restart_matches_edm_bit_for_bit(self):
        edm_net = CountingDenoiser()
        restart_net = CountingDenoiser()
        edm = generate.edm_sampler(
            edm_net, self.latents, num_steps=8, randn_like=seeded_randn_like(123))
        restart = generate.restart_sampler(
            restart_net, self.latents, num_steps=8, restart_info='{}',
            randn_like=seeded_randn_like(123))
        self.assertTrue(torch.equal(edm, restart))
        self.assertEqual(edm_net.num_calls, restart_net.num_calls)

    def test_network_calls_match_reported_nfe(self):
        config = '{"0":[3,2,0.06,0.30]}'
        net = CountingDenoiser()
        generate.restart_sampler(
            net, self.latents, num_steps=18, restart_info=config,
            randn_like=lambda x: torch.zeros_like(x))
        self.assertEqual(net.num_calls, generate.restart_nfe(18, config))
        self.assertEqual(net.num_calls, 43)

    def test_forward_jump_uses_variance_difference(self):
        config = '{"0":[3,2,0.06,0.30]}'
        net = CountingDenoiser(return_input=True)
        result = generate.restart_sampler(
            net, self.latents, num_steps=18, restart_info=config,
            randn_like=lambda x: torch.ones_like(x))

        indices = torch.arange(18, dtype=torch.float64)
        main_steps = (80 ** (1 / 7) + indices / 17 * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
        actual_t_min = main_steps[torch.argmin((main_steps - 0.06).abs())]
        expected_jump = 2 * (0.30 ** 2 - actual_t_min ** 2).sqrt()
        expected = self.latents.to(torch.float64) * 80 + expected_jump
        self.assertTrue(torch.allclose(result, expected, atol=1e-10))

    def test_rejects_levels_that_resolve_to_same_main_step(self):
        config = '{"a":[3,1,0.06,0.30],"b":[3,1,0.061,0.40]}'
        with self.assertRaisesRegex(ValueError, 'same main schedule index'):
            generate.restart_sampler(CountingDenoiser(), self.latents, restart_info=config)

    def test_restart_requires_denoiser_checkpoint(self):
        with self.assertRaises(click.ClickException):
            generate.check_sampler_compatibility('restart', VelocityNet(), {'num_steps': 18})

    def test_restart_options_are_not_silently_ignored(self):
        for sampler, net in [('edm', CountingDenoiser()), ('fm-euler', VelocityNet())]:
            with self.subTest(sampler=sampler), self.assertRaises(click.ClickException):
                generate.check_sampler_compatibility(
                    sampler, net, {'num_steps': 18, 'restart_info': '{}'})

    def test_help_exposes_restart_options(self):
        result = CliRunner().invoke(generate.main, ['--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('--sampler', result.output)
        self.assertIn('restart', result.output)
        self.assertIn('--restart_info', result.output)
        self.assertIn('--restart_noise', result.output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
