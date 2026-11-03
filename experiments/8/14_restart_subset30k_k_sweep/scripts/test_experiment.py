#!/usr/bin/env python3
"""CPU lifecycle checks with mocked generation/FID and temporary fixtures only."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
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
        self.sigmas = []

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma, labels):
        self.calls += 1
        self.sigmas.append(float(sigma))
        return x * 0.5


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = experiment.load_conditions(experiment.DEFAULT_ROOT / 'config' / experiment.CONFIG_NAME)
        for row in self.rows:
            model = self.root / Path(row['network']).name
            model.write_bytes(row['checkpoint'].encode())
            row['network'] = str(model)
        self.row = self.rows[1]
        reference = self.root / 'ref.npz'
        reference.write_bytes(b'reference fixture')
        self.args = SimpleNamespace(exp_root=self.root, fid_ref=reference, gen_batch=64,
                                    fid_batch=64, torchrun='torchrun', gpu_name='CPU mock GPU')

    def result(self, row, count=5000, fid=3.0):
        value = experiment.protocol(row, self.args)
        return dict(row, num_images=count, seeds=f'0-{count - 1}', fid=fid,
                    generate_sec=1000.0, fid_sec=20.0, reused_images=0, reuse_source='',
                    images='/pc-a/images', log='/pc-a/log', fid_ref='/pc-a/ref',
                    gen_batch=64, fid_batch=64, rho=7, restart_noise=1,
                    protocol_id=experiment.digest(value), protocol=value, gpu_name='CPU mock GPU',
                    generated_images=count, elapsed_sec=1030.0)

    def store(self, root, rows):
        for row in rows:
            experiment.write_json(root / 'results' / f"{row['condition']}-fid{row['num_images']}.json", row)

    def test_checkpoints_and_errors(self):
        for value, expected in [('', set(experiment.CHECKPOINTS)), ('20k-60k:40k', {'20k', '60k'}), ('40k', {'40k'}), ('80k,100k', {'80k', '100k'})]:
            self.assertEqual(experiment.parse_checkpoints(value), expected)
        for value in ('10k', '120k', '60k-20k', '20k-60k:0k', '20k-60k:30k'):
            with self.assertRaises(ValueError):
                experiment.parse_checkpoints(value)
        self.assertEqual(len(experiment.select_conditions(self.rows, '20k-60k', '')), 27)
        with self.assertRaises(ValueError):
            experiment.select_conditions(self.rows, '20k', '40k-k0')

    def test_all_conditions_and_actual_nfe_on_cpu(self):
        self.assertEqual(len(self.rows), 45)
        for row in self.rows[:9]:
            net = MockNet()
            torch.manual_seed(0)
            latent = torch.randn(2, 3, 2, 2)
            k = row['restart_k']
            if k:
                output = restart_sampler(net, latent, num_steps=row["main_steps"], restart_info={'0': [3, k, 0.14, 0.30]})
                self.assertEqual((row['actual_t_min'], row['restart_index']), experiment.schedule_point(row['main_steps']))
                self.assertLess(row['actual_t_min'], 0.30)
                restart_call = 2 * row['restart_index']
                self.assertAlmostEqual(net.sigmas[restart_call], 0.30, places=12)
                self.assertAlmostEqual(net.sigmas[restart_call - 1], row['actual_t_min'], places=12)
                self.assertAlmostEqual(net.sigmas[restart_call + 4 * k - 1], row['actual_t_min'], places=12)
            else:
                output = edm_sampler(net, latent, num_steps=row["main_steps"])
                self.assertIsNone(row['actual_t_min'])
            self.assertEqual(net.calls, row['nfe'])
            self.assertTrue(torch.isfinite(output).all())
            command = experiment.generate_command(row, self.root, '0-1', self.args)
            self.assertIn(f"--steps={row['main_steps']}", command)
            self.assertIn('--sampler=restart' if k else '--sampler=edm', command)

    def test_5k_to_25k_to_50k_resume_and_missing_seeds(self):
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
                seed_text = next(item.split('=', 1)[1] for item in command if item.startswith('--seeds='))
                new = set(experiment.parse_seeds(seed_text))
                self.assertFalse(new & present)
                present.update(new)
                return 2.0, None
            return 1.0, 7.0

        recovered = set()
        def recover(staging, destination, allowed):
            new = present - recovered
            recovered.update(present)
            return len(new)

        with mock.patch.object(experiment, 'recover_staging', side_effect=recover), \
             mock.patch.object(experiment, 'inventory', side_effect=inventory), \
             mock.patch.object(experiment, 'link_image'), \
             mock.patch.object(experiment, 'run_logged', side_effect=run), \
             mock.patch.object(experiment, 'aggregate'):
            results = []
            with self.assertRaisesRegex(ValueError, 'extension requires'):
                experiment.evaluate(self.row, 50000, self.args, results)
            experiment.evaluate(self.row, 5000, self.args, results)
            # Simulate an interrupted extension: only missing ranges should generate.
            present.update(range(5000, 6000))
            experiment.evaluate(self.row, 25000, self.args, results)
            experiment.evaluate(self.row, 50000, self.args, results)
            self.assertEqual(len(commands), 6)
            self.assertIn('--seeds=6000-24999', commands[2])
            self.assertIn('--seeds=25000-49999', commands[4])
            self.assertEqual(results[-1]['generated_images'], 25000)
            experiment.validate_results(results)
            for count in experiment.STAGES:
                experiment.evaluate(self.row, count, self.args, results)
            self.assertEqual(len(commands), 6)
            present.remove(49999)
            with self.assertRaises(AssertionError):
                experiment.evaluate(self.row, 50000, self.args, results)
            self.args.gen_batch = 32
            with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
                experiment.evaluate(self.row, 50000, self.args, results)

    def test_aggregation_partial_and_tie(self):
        results = [self.result(row) for row in self.rows[:9]]
        experiment.aggregate(self.root, results[:8])
        self.assertEqual(experiment.read_tsv(self.root / 'results/optimal_k.tsv'), [])
        experiment.aggregate(self.root, results)
        optimal = experiment.read_tsv(self.root / 'results/optimal_k.tsv')
        self.assertEqual(optimal[0]['optimal_k'], '0')
        self.assertEqual(optimal[0]['candidate_ks'], '0,1,2,3,4,5,6,7,8')
        self.assertTrue((self.root / f'results/fixed_{experiment.MODE}_tmin014_fid5000.svg').is_file())

    def test_merge_json_only_and_mismatches(self):
        a, b = self.root / 'pc-a', self.root / 'pc-b'
        first = self.result(self.rows[0])
        second = self.result(self.rows[9])
        second.update(network='/pc-b/model', images='/pc-b/images', log='/pc-b/log',
                      fid_ref='/pc-b/ref', gpu_name='Different GPU')
        self.store(a, [first])
        self.store(b, [second])
        self.assertEqual(len(experiment.merged_results([a, b])), 2)
        # Aggregation must need neither source models nor images, including via CLI.
        for path in self.root.glob('network-snapshot-*'):
            path.unlink()
        command = [str(experiment.SCRIPT_ROOT / 'run.sh'), '--aggregate-only', '--result-roots', str(a), str(b),
                   '--exp-root', str(self.root / 'merged'), '--fid-ref', str(self.root / 'missing-ref')]
        env = dict(os.environ, PYTHON=sys.executable, EDM_GPU_CONFIG=str(self.root / 'missing-gpu'))
        output = subprocess.check_output(command, env=env, text=True)
        self.assertIn('Aggregated 2', output)
        self.assertEqual(len(json.loads((self.root / 'merged/results/restart_fid_summary.json').read_text())), 2)
        for field in ('fid_ref', 'generate', 'fid', 'runner'):
            changed = copy.deepcopy(second)
            changed['protocol'][field]['sha256'] = 'a' * 64
            changed['protocol_id'] = experiment.digest(changed['protocol'])
            self.store(b, [changed])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'disagree'):
                experiment.merged_results([a, b])
        changed = copy.deepcopy(second)
        changed['gen_batch'] = changed['protocol']['gen_batch'] = 32
        changed['protocol_id'] = experiment.digest(changed['protocol'])
        self.store(b, [changed])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            experiment.merged_results([a, b])
        self.store(b, [second, first])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            experiment.merged_results([a, b])

    def test_model_seed_condition_mismatch(self):
        first, second = self.result(self.rows[0]), self.result(self.rows[1])
        second['protocol']['network']['sha256'] = 'a' * 64
        second['protocol_id'] = experiment.digest(second['protocol'])
        with self.assertRaisesRegex(ValueError, 'Model SHA256'):
            experiment.validate_results([first, second])
        for key, value in (('seeds', '1-5000'), ('nfe', 999), ('fid', float('nan'))):
            changed = copy.deepcopy(first)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.validate_results([changed])

    def test_portable_protocol_and_content_change(self):
        old = experiment.protocol(self.row, self.args)
        moved = self.root / 'relocated-model'
        moved.write_bytes(Path(self.row['network']).read_bytes())
        row = dict(self.row, network=str(moved))
        self.assertEqual(old, experiment.protocol(row, self.args))
        experiment.ensure_protocol(self.root / 'pool', old)
        moved.write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
            experiment.ensure_protocol(self.root / 'pool', experiment.protocol(row, self.args))

    def test_plan_selected_models_only_env_precedence_and_gpu_bypass(self):
        config = self.root / 'config.tsv'
        fields = ['condition', 'checkpoint', 'main_steps', 'restart_steps', 'restart_k', 't_min', 't_max', 'nfe', 'network']
        experiment.write_tsv(config, [{key: row[key] for key in fields} for row in self.rows], fields)
        for path in self.root.glob('network-snapshot-*'):
            if path.name != 'network-snapshot-020000.pkl':
                path.unlink()
        env = dict(os.environ, PYTHON=sys.executable, CHECKPOINT_FILTER='40k-60k',
                   EDM_GPU_CONFIG=str(self.root / 'missing-gpu.env'))
        command = [str(experiment.SCRIPT_ROOT / 'run.sh'), '--plan-only', '--fid-stage', 'all',
                   '--checkpoint-filter', '20k', '--exp-root', str(self.root / 'plan'),
                   '--config', str(config), '--fid-ref', str(self.args.fid_ref)]
        output = subprocess.check_output(command, env=env, text=True)
        self.assertIn('conditions=9; stages=[5000, 25000, 50000]', output)
        self.assertIn('new_images=450000; pending FID evaluations=27', output)
        self.assertFalse((self.root / 'plan').exists())
        command[command.index('20k')] = '40k'
        proc = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('network-snapshot-040000.pkl', proc.stderr)
        proc = subprocess.run([str(experiment.SCRIPT_ROOT / 'run.sh'), '--samples-only'],
                              env=env, text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Missing local GPU configuration', proc.stderr)

    def test_png_corruption_extra_seed_and_protocol_guard(self):
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
        with self.assertRaisesRegex(ValueError, 'Unexpected seed'):
            experiment.inventory(pool, [])
        target.write_bytes(b'corrupt')
        with self.assertRaises(OSError):
            experiment.inventory(pool, range(1))

    def test_remaining_time_after_complete_pilot(self):
        selected = self.rows[:18]
        pilot = [self.result(row) for row in self.rows[:9]]
        experiment.update_estimate(self.args, selected, experiment.STAGES, pilot[:8])
        self.assertFalse((self.root / 'results/time_estimate.json').exists())
        experiment.update_estimate(self.args, selected, experiment.STAGES, pilot)
        value = json.loads((self.root / 'results/time_estimate.json').read_text())
        self.assertEqual(value['pilot_checkpoint'], '20k')
        self.assertEqual([stage['pending_evaluations'] for stage in value['stages']], [9, 18, 18])
        self.assertGreater(value['remaining_sec'], 0)
        completed = [self.result(row, count) for row in selected for count in experiment.STAGES]
        experiment.update_estimate(self.args, selected, experiment.STAGES, completed)
        value = json.loads((self.root / 'results/time_estimate.json').read_text())
        self.assertEqual(value['remaining_sec'], 0)


    def test_staging_recovery_preserves_pool_and_discards_partial_png(self):
        from PIL import Image
        staging, pool = self.root / 'staging', self.root / 'pool'
        for seed in (0, 1, 2):
            path = staging / experiment.image_relative(seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            if seed == 1:
                path.write_bytes(b'incomplete PNG')
            else:
                Image.new('RGB', (32, 32)).save(path)
        self.assertEqual(experiment.recover_staging(staging, pool, range(3)), 2)
        self.assertEqual(set(experiment.inventory(pool, range(3))), {0, 2})
        self.assertEqual(list(staging.rglob('*.png')), [])
        source = staging / experiment.image_relative(0)
        Image.new('RGB', (32, 32), 'red').save(source)
        original = (pool / experiment.image_relative(0)).read_bytes()
        with self.assertRaisesRegex(ValueError, 'differs'):
            experiment.recover_staging(staging, pool, range(3))
        self.assertEqual((pool / experiment.image_relative(0)).read_bytes(), original)

    def test_pause_during_generation_finishes_all_missing_ranges(self):
        from PIL import Image
        pool = self.root / 'images' / self.row['condition'] / 'pool'
        value = experiment.protocol(self.row, self.args)
        experiment.ensure_protocol(pool, value)
        target = pool / experiment.image_relative(1)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 32)).save(target)
        self.args.pause_file = self.root / 'PAUSE'
        progress_path = self.root / 'progress/generation.json'
        progress = experiment.progress_record(progress_path, value)
        commands = []
        def run(command, log):
            self.args.pause_file.touch()
            commands.append(command)
            destination = Path(next(x.split('=', 1)[1] for x in command if x.startswith('--outdir=')))
            seeds = experiment.parse_seeds(next(x.split('=', 1)[1] for x in command if x.startswith('--seeds=')))
            for seed in seeds:
                target = destination / experiment.image_relative(seed)
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (32, 32)).save(target)
            return 0.5, None
        with mock.patch.object(experiment, 'run_logged', side_effect=run):
            experiment.generate_missing(self.row, pool, range(3), self.args, self.root / 'generate.log',
                                        progress, progress_path, value)
        self.assertEqual(len(commands), 2)
        self.assertIn('--seeds=0-0', commands[0])
        self.assertIn('--seeds=2-2', commands[1])
        self.assertEqual(set(experiment.inventory(pool, range(3), complete=True)), {0, 1, 2})

    def test_run_jobs_skips_completed_and_does_not_wait_after_final_result(self):
        self.args.pause_file = self.root / 'PAUSE'
        self.args.pause_poll_sec = 0.02
        self.args.samples_only = False
        rows = self.rows[:2]
        controller = mock.Mock()
        controller.info = {}
        order = []
        def boundary(row, seeds):
            order.append(('boundary', row['condition']))
            if self.args.pause_file.exists():
                self.assertEqual(order[-2], ('finished', rows[0]['condition']))
                self.args.pause_file.unlink()
        def evaluate(row, count, args, results):
            self.assertFalse(args.pause_file.exists())
            args.pause_file.touch()
            order.append(('finished', row['condition']))
        controller.boundary.side_effect = boundary
        with mock.patch.dict(os.environ), \
             mock.patch.object(experiment, 'ConditionPause', return_value=controller) as factory, \
             mock.patch.object(experiment, 'evaluate', side_effect=evaluate), \
             mock.patch.object(experiment, 'update_estimate'):
            experiment.run_jobs(self.args, rows, [5000], [])
        self.assertEqual(controller.boundary.call_count, 2)
        controller.close.assert_called_once_with('complete')
        self.assertTrue(self.args.pause_file.exists())
        self.assertEqual(controller.info['num_images'], 5000)
        self.assertEqual(factory.call_args.args[1], experiment.EXPERIMENT_ID)
        completed = [self.result(row) for row in rows]
        with mock.patch.object(experiment, 'ConditionPause') as factory, \
             mock.patch.object(experiment, 'evaluate') as verify:
            experiment.run_jobs(self.args, rows, [5000], completed)
        factory.assert_not_called()
        self.assertEqual(verify.call_count, 2)

    def test_running_command_is_not_terminated_by_pause_file(self):
        self.args.pause_file = self.root / 'PAUSE'
        self.args.pause_poll_sec = 0.02
        done = self.root / 'finished'
        code = (f'import pathlib,time; pathlib.Path({str(self.args.pause_file)!r}).touch(); '
                f'time.sleep(0.2); pathlib.Path({str(done)!r}).write_text("done"); print(4.0)')
        with mock.patch.object(experiment, 'terminate_group') as terminate:
            _, scalar = experiment.run_logged([sys.executable, '-c', code], self.root / 'child.log')
        self.assertEqual(scalar, 4.0)
        self.assertTrue(done.exists())
        terminate.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'Command failed'):
            experiment.run_logged([sys.executable, '-c', 'raise SystemExit(2)'], self.root / 'failure.log')

    def test_pause_after_generation_allows_fid_and_result_before_waiting(self):
        pool = self.root / 'images' / self.row['condition'] / 'pool'
        (self.root / 'logs').mkdir()
        marker = self.root / 'results' / f"{self.row['condition']}-fid5000.json"
        self.args.pause_file = self.root / 'PAUSE'
        self.args.samples_only = False
        calls = []
        def inventory(directory, allowed, **kwargs):
            return {seed: pool / experiment.image_relative(seed) for seed in allowed}
        def generate(*args):
            self.args.pause_file.touch()
            calls.append('generate')
        def run(command, log):
            self.assertTrue(self.args.pause_file.exists())
            self.assertFalse(marker.exists())
            calls.append('fid')
            return 0.3, 4.0
        controller = mock.Mock()
        controller.info = {}
        controller.boundary.side_effect = lambda *args: calls.append('boundary')
        with mock.patch.object(experiment, 'inventory', side_effect=inventory), \
             mock.patch.object(experiment, 'generate_missing', side_effect=generate), \
             mock.patch.object(experiment, 'link_image'), \
             mock.patch.object(experiment, 'run_logged', side_effect=run), \
             mock.patch.object(experiment, 'aggregate'):
            results = []
            experiment.run_condition(self.row, 5000, self.args, results, controller)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(experiment.load_results(self.root)), 1)
        self.assertEqual(calls, ['boundary', 'generate', 'fid'])
        controller.boundary.assert_called_once()
        controller.phase.assert_any_call('saved')
        self.assertTrue(self.args.pause_file.exists())

    def test_complete_plan_and_split_counts_without_writes(self):
        config = self.root / 'config.tsv'
        fields = ['condition', 'checkpoint', 'main_steps', 'restart_steps', 'restart_k', 't_min', 't_max', 'nfe', 'network']
        experiment.write_tsv(config, [{key: row[key] for key in fields} for row in self.rows], fields)
        output_root = self.root / 'no-writes'
        env = dict(os.environ, PYTHON=sys.executable, EDM_GPU_CONFIG=str(self.root / 'missing-gpu'))
        command = [str(experiment.SCRIPT_ROOT / 'run.sh'), '--plan-only', '--fid-stage', 'all',
                   '--config', str(config), '--fid-ref', str(self.args.fid_ref), '--exp-root', str(output_root),
                   '--pause-file', str(self.root / 'PAUSE')]
        (self.root / 'PAUSE').touch()
        for selection, conditions, evaluations, images in [('all', 45, 135, 2250000), ('20k-40k', 18, 54, 900000), ('60k', 9, 27, 450000), ('80k,100k', 18, 54, 900000)]:
            output = subprocess.check_output(command + ['--checkpoint-filter', selection], env=env, text=True)
            self.assertIn(f'conditions={conditions}', output)
            self.assertIn(f'new_images={images}; pending FID evaluations={evaluations}', output)
            self.assertFalse(output_root.exists())


    def test_atomic_save_failure_retains_previous_json(self):
        path = self.root / 'saved.json'
        experiment.write_json(path, {'state': 'complete'})
        with mock.patch.object(Path, 'replace', side_effect=OSError('interrupted before replace')):
            with self.assertRaises(OSError):
                experiment.write_json(path, {'state': 'new'})
        self.assertEqual(json.loads(path.read_text()), {'state': 'complete'})
        experiment.write_json(path, {'state': 'recovered'})
        self.assertEqual(json.loads(path.read_text()), {'state': 'recovered'})

    def test_invalid_configuration_is_rejected(self):
        fields = ['condition', 'checkpoint', 'main_steps', 'restart_steps', 'restart_k', 't_min', 't_max', 'nfe', 'network']
        original = [{key: row[key] for key in fields} for row in self.rows]
        config = self.root / 'invalid.tsv'
        for key, value in [('main_steps', 19), ('nfe', 51), ('t_min', 0.06)]:
            rows = copy.deepcopy(original)
            rows[1][key] = value
            experiment.write_tsv(config, rows, fields)
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.load_conditions(config)
        experiment.write_tsv(config, original[:-1], fields)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            experiment.load_conditions(config)


if __name__ == '__main__':
    unittest.main()
