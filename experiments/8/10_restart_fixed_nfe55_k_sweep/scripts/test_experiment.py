#!/usr/bin/env python3
"""CPU checks for fixed-NFE sampling and experiment lifecycle; no real FID."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_experiment as experiment

sys.path.insert(0, str(experiment.REPO_ROOT))
import torch
from generate import edm_sampler, restart_sampler


class MockNet:
    sigma_min = 0
    sigma_max = float('inf')

    def __init__(self):
        self.calls = 0

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma, labels):
        self.calls += 1
        return x * 0.5


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = experiment.load_conditions(experiment.DEFAULT_ROOT / 'config/restart_fixed_nfe55.tsv')
        self.row = next(row for row in self.rows if row['restart_k'] == 1)
        self.args = SimpleNamespace(exp_root=self.root, fid_ref=experiment.REPO_ROOT / 'fid-refs/cifar10-32x32.npz',
                                    gen_batch=64, fid_batch=64, torchrun='torchrun')

    def test_all_80_conditions_and_actual_nfe(self):
        self.assertEqual(len(self.rows), 80)
        self.assertNotIn(5, {row['restart_k'] for row in self.rows})
        for row in self.rows[:8]:
            net = MockNet()
            torch.manual_seed(0)
            latent = torch.randn(2, 3, 2, 2)
            k = row['restart_k']
            if k:
                output = restart_sampler(net, latent, num_steps=row['main_steps'], restart_info={'0': [3, k, 0.06, 0.30]})
                times = (80 ** (1 / 7) + torch.arange(row['main_steps'], dtype=torch.float64)
                         / (row['main_steps'] - 1) * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
                index = int(torch.argmin((torch.cat([times, torch.zeros(1)]) - 0.06).abs()))
                self.assertEqual(index, row['restart_index'])
                self.assertAlmostEqual(float(times[index]), row['actual_t_min'], places=12)
            else:
                output = edm_sampler(net, latent, num_steps=row['main_steps'])
                self.assertIsNone(row['actual_t_min'])
            self.assertEqual(net.calls, 55)
            self.assertTrue(torch.isfinite(output).all())
            command = experiment.generate_command(row, self.root, '0-1', self.args)
            self.assertIn(f"--steps={row['main_steps']}", command)
            self.assertIn('--sampler=restart' if k else '--sampler=edm', command)

    def test_5k_to_25k_and_resume(self):
        present, commands = set(), []
        pool = self.root / 'images' / self.row['condition'] / 'pool'
        (self.root / 'logs').mkdir()

        def inventory(directory, allowed, complete=False, verify=True):
            seeds = present & set(allowed)
            if complete:
                self.assertEqual(seeds, set(allowed))
            return {seed: pool / experiment.image_relative(seed) for seed in seeds}

        def run(command, log):
            commands.append(command)
            if 'generate.py' in command:
                seed_text = next(value.split('=', 1)[1] for value in command if value.startswith('--seeds='))
                new = set(experiment.parse_seeds(seed_text))
                self.assertFalse(new & present)
                present.update(new)
                return 2.0, None
            return 1.0, 7.0

        with mock.patch.object(experiment, 'inventory', side_effect=inventory), \
             mock.patch.object(experiment, 'link_image'), \
             mock.patch.object(experiment, 'run_logged', side_effect=run), \
             mock.patch.object(experiment, 'aggregate'):
            results = []
            experiment.evaluate(self.row, 5000, self.args, results)
            experiment.evaluate(self.row, 25000, self.args, results)
            self.assertEqual(len(commands), 4)
            self.assertIn('--seeds=5000-24999', commands[2])
            experiment.evaluate(self.row, 25000, self.args, results)
            self.assertEqual(len(commands), 4)
            self.args.gen_batch = 32
            with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
                experiment.evaluate(self.row, 25000, self.args, results)

    def test_partial_aggregation_and_tie(self):
        results = [dict(row, num_images=5000, seeds='0-4999', fid=3.0,
                        generate_sec=0, fid_sec=0, reused_images=0, reuse_source='', images='', log='',
                        fid_ref='', gen_batch=64, fid_batch=64, rho=7, restart_noise=1, protocol_id='test')
                   for row in self.rows[:8]]
        experiment.aggregate(self.root, results[:7])
        self.assertEqual(experiment.read_tsv(self.root / 'results/optimal_k.tsv'), [])
        experiment.aggregate(self.root, results)
        optimal = experiment.read_tsv(self.root / 'results/optimal_k.tsv')
        self.assertEqual(optimal[0]['optimal_k'], '0')
        self.assertEqual(optimal[0]['candidate_ks'], '0,1,2,3,4,6,7,8')
        self.assertTrue((self.root / 'results/fixed_nfe55_fid5000.svg').is_file())

    def test_plan_only_writes_nothing_and_ignores_gpu_config(self):
        env = dict(os.environ, EDM_GPU_CONFIG=str(self.root / 'missing-gpu.env'), PYTHON=sys.executable)
        command = [str(experiment.SCRIPT_ROOT / 'run.sh'), '--plan-only', '--fid-stage', 'all', '--exp-root', str(self.root / 'plan')]
        output = subprocess.check_output(command, env=env, text=True)
        self.assertIn('conditions=80; stages=[5000, 25000]', output)
        self.assertIn('new_images=2000000', output)
        self.assertFalse((self.root / 'plan').exists())
        proc = subprocess.run([str(experiment.SCRIPT_ROOT / 'run.sh'), '--samples-only'], env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Missing local GPU configuration', proc.stderr)

    def test_png_inventory_and_protocol_guard(self):
        from PIL import Image
        pool = self.root / 'pool'
        value = experiment.protocol(self.row, self.args)
        experiment.ensure_protocol(pool, value, create=False)
        self.assertFalse(pool.exists())
        experiment.ensure_protocol(pool, value)
        target = pool / experiment.image_relative(0)
        target.parent.mkdir(exist_ok=True)
        Image.new('RGB', (32, 32)).save(target)
        self.assertEqual(set(experiment.inventory(pool, range(1), complete=True)), {0})
        target.write_bytes(b'corrupt')
        with self.assertRaises(OSError):
            experiment.inventory(pool, range(1))


if __name__ == '__main__':
    unittest.main()
