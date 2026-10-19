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
import signal
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
        self.rows = experiment.load_conditions(experiment.DEFAULT_ROOT / 'config/restart_fixed_nfe55_tmin014.tsv')
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
        cases = {'': set(experiment.CHECKPOINTS), 'all': set(experiment.CHECKPOINTS),
                 '40k': {'40k'}, '20k,60k 100k 20k': {'20k', '60k', '100k'},
                 '20k-140k': {f'{n}k' for n in range(20, 141, 20)},
                 '20k-140k:40k': {'20k', '60k', '100k', '140k'}}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(experiment.parse_checkpoints(value), expected)
        for value in ('10k', '220k', '140k-20k', '20k-140k:0k', '20k-140k:30k',
                      '20k-140k:80k', 'all,20k', '20', '20k:40k'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                experiment.parse_checkpoints(value)
        self.assertEqual(len(experiment.select_conditions(self.rows, '20k-120k', '')), 54)
        self.assertEqual(len(experiment.select_conditions(self.rows, '140k-200k', '')), 36)
        with self.assertRaises(ValueError):
            experiment.select_conditions(self.rows, '20k', '160k-k0')

    def test_all_90_conditions_and_actual_nfe_on_cpu(self):
        self.assertEqual(len(self.rows), 90)
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
            self.assertEqual(net.calls, 55)
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
        self.assertTrue((self.root / 'results/fixed_nfe55_tmin014_fid5000.svg').is_file())

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
        env = dict(os.environ, PYTHON=sys.executable, CHECKPOINT_FILTER='160k-200k',
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

    def test_interrupted_generation_recovers_then_retries_missing_only(self):
        from PIL import Image
        pool = self.root / 'images' / self.row['condition'] / 'pool'
        value = experiment.protocol(self.row, self.args)
        experiment.ensure_protocol(pool, value)
        progress_path = self.root / 'progress/generation.json'
        progress = experiment.progress_record(progress_path, value)
        log = self.root / 'generation.log'
        commands = []
        def run(command, log):
            commands.append(command)
            destination = Path(next(x.split('=', 1)[1] for x in command if x.startswith('--outdir=')))
            seeds = experiment.parse_seeds(next(x.split('=', 1)[1] for x in command if x.startswith('--seeds=')))
            for seed in seeds:
                target = destination / experiment.image_relative(seed)
                target.parent.mkdir(parents=True, exist_ok=True)
                if len(commands) == 1 and seed == 1:
                    target.write_bytes(b'partial')
                    raise experiment.PausedCommand(0.25)
                Image.new('RGB', (32, 32)).save(target)
            return 0.5, None
        with mock.patch.object(experiment, 'run_logged', side_effect=run), \
             mock.patch.object(experiment, 'wait_for_permission') as wait:
            experiment.generate_missing(self.row, pool, range(3), self.args, log,
                                        progress, progress_path, value)
        self.assertEqual(len(commands), 2)
        self.assertIn('--seeds=1-2', commands[1])
        self.assertEqual(progress['generated_images'], 3)
        self.assertEqual(progress['generate_sec'], 0.75)
        self.assertEqual(set(experiment.inventory(pool, range(3), complete=True)), {0, 1, 2})
        wait.assert_called_once()

    def test_pause_before_launch_waits_and_rechecks_gpu(self):
        self.args.pause_file = self.root / 'PAUSE'
        self.args.pause_poll_sec = 0.02
        self.args.pause_file.touch()
        def remove():
            time.sleep(0.1)
            self.args.pause_file.unlink()
        worker = threading.Thread(target=remove)
        worker.start()
        def waiter(*args, **kwargs):
            self.assertFalse(self.args.pause_file.exists())
        with mock.patch.object(experiment.subprocess, 'run', side_effect=waiter) as shared, \
             mock.patch.object(experiment.subprocess, 'check_output', side_effect=['0', 'CPU mock GPU']), \
             mock.patch.dict(os.environ):
            experiment.wait_for_permission(self.args, initial=True)
        worker.join()
        shared.assert_called_once()
        self.assertTrue(shared.call_args[0][0][0].endswith('wait_for_gpu_and_run.sh'))

    def test_running_process_group_pause_and_force_kill(self):
        self.args.pause_file = self.root / 'PAUSE'
        self.args.pause_poll_sec = 0.02
        for ignore in (False, True):
            ready = self.root / f'ready-{ignore}'
            code = ('import signal,time,pathlib,subprocess,sys; '
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore else
                    'import time,pathlib,subprocess,sys; ')
            # The child shares the process group, including the ignore-TERM case.
            code += (f'subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                     f'pathlib.Path({str(ready)!r}).write_text("ready"); time.sleep(30)')
            def pause():
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.args.pause_file.touch()
            worker = threading.Thread(target=pause)
            worker.start()
            started = time.monotonic()
            with mock.patch.object(experiment, 'ACTIVE_ARGS', self.args), \
                 mock.patch.object(experiment, 'wait_for_permission'), \
                 mock.patch.object(experiment, 'TERMINATE_GRACE_SEC', 0.2):
                with self.assertRaises(experiment.PausedCommand):
                    experiment.run_logged([sys.executable, '-c', code], self.root / 'child.log')
            worker.join()
            self.assertLess(time.monotonic() - started, 3)
            self.assertTrue(self.args.needs_gpu_wait)
            self.args.pause_file.unlink()
        # Non-pause failures must surface, rather than automatically retry.
        with self.assertRaisesRegex(RuntimeError, 'Command failed'):
            experiment.run_logged([sys.executable, '-c', 'raise SystemExit(2)'], self.root / 'failure.log')

    def test_fid_interruption_keeps_images_and_creates_marker_only_after_success(self):
        pool = self.root / 'images' / self.row['condition'] / 'pool'
        (self.root / 'logs').mkdir()
        marker = self.root / 'results' / f"{self.row['condition']}-fid5000.json"
        calls = []
        def inventory(directory, allowed, **kwargs):
            return {seed: pool / experiment.image_relative(seed) for seed in allowed}
        def run(command, log):
            self.assertFalse(marker.exists())
            calls.append(command)
            if len(calls) == 1:
                raise experiment.PausedCommand(0.2)
            return 0.3, 4.0
        with mock.patch.object(experiment, 'inventory', side_effect=inventory), \
             mock.patch.object(experiment, 'generate_missing') as generate, \
             mock.patch.object(experiment, 'link_image'), \
             mock.patch.object(experiment, 'run_logged', side_effect=run), \
             mock.patch.object(experiment, 'wait_for_permission') as wait, \
             mock.patch.object(experiment, 'aggregate'):
            results = []
            experiment.evaluate(self.row, 5000, self.args, results)
        self.assertEqual(len(calls), 2)
        wait.assert_called_once()
        self.assertEqual(results[0]['fid_sec'], 0.5)
        self.assertEqual(json.loads(marker.read_text())['fid'], 4.0)
        # An interrupted aggregate after atomic completion still leaves a resumable marker.
        self.assertEqual(len(experiment.load_results(self.root)), 1)

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
        for selection, conditions, evaluations, images in [('all', 90, 270, 4500000),
                                                         ('20k-120k', 54, 162, 2700000),
                                                         ('140k-200k', 36, 108, 1800000)]:
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
        for key, value in [('main_steps', 18), ('nfe', 51), ('t_min', 0.06)]:
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
