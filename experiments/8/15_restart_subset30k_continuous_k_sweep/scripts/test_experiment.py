#!/usr/bin/env python3
"""CPU-only regression tests for snapshot watching, direct FID, restart and pause."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import watch_checkpoints as watch


class ProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'snapshot.pkl'
        self.now = 0
        self.validate = mock.Mock()
        self.probe = watch.SnapshotProbe(validate=self.validate, clock=lambda: self.now)

    def test_missing_arrival_writing_stability_and_cache(self):
        self.assertFalse(self.probe.ready(self.path))
        self.path.write_bytes(b'part')
        self.assertFalse(self.probe.ready(self.path))
        self.now = 29
        self.assertFalse(self.probe.ready(self.path))
        self.path.write_bytes(b'complete')
        self.assertFalse(self.probe.ready(self.path))
        self.now = 58
        self.assertFalse(self.probe.ready(self.path))
        self.now = 59
        self.assertTrue(self.probe.ready(self.path))
        self.assertTrue(self.probe.ready(self.path))
        self.validate.assert_called_once_with(self.path)

    def test_changed_during_load_retries_but_stable_corruption_fails(self):
        self.path.write_bytes(b'part')
        self.probe.ready(self.path)
        self.now = 30
        def changing(path):
            path.write_bytes(b'changed')
            raise ValueError('partial')
        self.validate.side_effect = changing
        self.assertFalse(self.probe.ready(self.path))
        self.probe.ready(self.path)
        self.now = 60
        self.validate.side_effect = ValueError('corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.probe.ready(self.path)

    def test_cpu_loader_hides_cuda(self):
        with mock.patch.object(watch.subprocess, 'run') as run:
            watch.cpu_validate(self.path)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(run.call_args.kwargs['check'])


class EngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.module = watch.engine(os.environ.get('FIXED_MODE', 'main18'))
        self.rows = self.module.load_conditions(self.module.DEFAULT_ROOT / 'config' / self.module.CONFIG_NAME)
        for row in self.rows:
            path = self.root / Path(row['network']).name
            path.write_bytes(row['checkpoint'].encode())
            row['network'] = str(path)
        ref = self.root / 'ref.npz'
        ref.write_bytes(b'reference')
        self.args = SimpleNamespace(exp_root=self.root, fid_ref=ref, gen_batch=64, fid_batch=64,
                                    torchrun='torchrun', gpu_name='CPU mock', pause=None,
                                    samples_only=False, sample_seeds='0-1')

    def test_direct_50k_restart_missing_images_protocol_and_failure(self):
        module, row = self.module, self.rows[0]
        present, commands = set(range(123)), []
        pool = self.root / 'images' / row['condition'] / 'pool'
        (self.root / 'logs').mkdir()
        def inventory(directory, allowed, complete=False, verify=True):
            selected = present & set(allowed)
            if complete and selected != set(allowed):
                raise ValueError('missing images')
            return {seed: pool / module.image_relative(seed) for seed in selected}
        def run(command, log):
            commands.append(command)
            if 'generate.py' in command:
                seeds = next(s.split('=', 1)[1] for s in command if s.startswith('--seeds='))
                present.update(module.parse_seeds(seeds))
                return 2, None
            return 1, 3.5
        with mock.patch.object(module, 'inventory', side_effect=inventory), \
             mock.patch.object(module, 'recover_staging', return_value=0), \
             mock.patch.object(module, 'link_image'), \
             mock.patch.object(module, 'run_logged', side_effect=run), \
             mock.patch.object(module, 'aggregate'):
            results = []
            module.evaluate(row, 50000, self.args, results)
            self.assertEqual(len(commands), 2)
            self.assertIn('--seeds=123-49999', commands[0])
            self.assertIn('--num=50000', commands[1])
            module.validate_results(results)
            self.assertEqual(len(module.load_results(self.root)), 1)
            evaluation = watch.Evaluation(module, self.args, [row])
            self.assertTrue(evaluation.completed([row]))
            module.evaluate(row, 50000, self.args, results)
            self.assertEqual(len(commands), 2)
            present.remove(49999)
            with self.assertRaisesRegex(ValueError, 'missing images'):
                evaluation.completed([row])
            with self.assertRaisesRegex(ValueError, 'missing images'):
                module.evaluate(row, 50000, self.args, results)
            present.add(49999)
            self.args.gen_batch = 32
            with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
                module.evaluate(row, 50000, self.args, results)
            self.args.gen_batch = 64
            changed = copy.deepcopy(results)
            changed[0]['protocol']['network']['sha256'] = 'a' * 64
            with self.assertRaises(ValueError):
                module.validate_results(changed)

    def test_stop_during_generation_finishes_fid_save_and_defers_next(self):
        module, row = self.module, self.rows[0]
        self.args.snapshot_signature = watch.signature(Path(row['network']))
        controller = mock.Mock(info={})
        events = []
        def evaluate(row, count, args, results):
            events.append('generate')
            (self.root / 'STOP').touch()
            events.extend(['fid', 'save'])
        controller.boundary.side_effect = lambda *a: events.append('boundary')
        with mock.patch.object(module, 'evaluate', side_effect=evaluate):
            module.run_condition(row, 50000, self.args, [], controller)
        self.assertEqual(events, ['boundary', 'generate', 'fid', 'save'])
        self.assertEqual(controller.boundary.call_count, 1)
        controller.phase.assert_called_with('saved')

    def test_snapshot_changed_during_gpu_wait_is_rejected(self):
        row = self.rows[0]
        self.args.snapshot_signature = watch.signature(Path(row['network']))
        controller = mock.Mock(info={})
        controller.boundary.side_effect = lambda *a: Path(row['network']).write_bytes(b'new')
        with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
            self.module.run_condition(row, 50000, self.args, [], controller)

    def test_real_pause_after_saved_condition_and_gpu_recheck(self):
        stop = self.root / 'PAUSE'
        bindir = self.root / 'bin'
        bindir.mkdir()
        gpu = bindir / 'nvidia-smi'
        gpu.write_text('#!/usr/bin/env bash\nif [[ "$*" == *"--query-gpu=name"* ]]; then echo "Mock GPU"; fi\n')
        gpu.chmod(0o755)
        config = self.root / 'gpu.env'
        config.write_text('GPU_INDEX=7\nEXPECTED_GPU_NAME="Mock GPU"\n')
        env = dict(PATH=f'{bindir}:{os.environ["PATH"]}', EDM_GPU_CONFIG=str(config),
                   PAUSE_FILE=str(stop), PAUSE_ENABLED='1', PAUSE_POLL_SEC='0.01',
                   PAUSE_ETA_UPDATE_SEC='0.02', CHECK_INTERVAL_SEC='1')
        with mock.patch.dict(os.environ, env):
            pause = watch.ExperimentPause(self.root / 'run', '8-15-test', self.root / 'summary',
                                           'ref', overall_root=self.root)
            self.addCleanup(pause.close)
            rows = self.rows[:2]
            self.args.snapshot_signature = watch.signature(Path(rows[0]['network']))
            events, errors = [], []
            def evaluate(row, count, args, results):
                events.append('generate')
                if len(events) == 1:
                    stop.touch()
                events.extend(['fid', 'saved'])
            with mock.patch.object(self.module, 'evaluate', side_effect=evaluate):
                self.module.run_condition(rows[0], 50000, self.args, [], pause)
                self.assertEqual(events, ['generate', 'fid', 'saved'])
                def next_condition():
                    try:
                        self.module.run_condition(rows[1], 50000, self.args, [], pause)
                    except BaseException as error:
                        errors.append(error)
                thread = threading.Thread(target=next_condition, daemon=True)
                thread.start()
                try:
                    deadline = time.monotonic() + 8
                    log = self.root / 'run/logs/pause.log'
                    while 'PAUSE before' not in log.read_text():
                        if time.monotonic() > deadline:
                            self.fail('No condition-boundary pause')
                        time.sleep(.01)
                    self.assertEqual(events, ['generate', 'fid', 'saved'])
                    stop.unlink()
                    thread.join(8)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(events, ['generate', 'fid', 'saved'] * 2)
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')
                    self.assertEqual(log.read_text().count('RESUME GPU free'), 2)
                finally:
                    if thread.is_alive():
                        pause.stop_file.touch()
                        thread.join(8)

    def test_configuration_sampler_nfe_on_cpu(self):
        sys.path.insert(0, str(watch.REPO))
        import torch
        from generate import edm_sampler, restart_sampler
        class Net:
            sigma_min, sigma_max = 0, float('inf')
            calls = 0
            def round_sigma(self, sigma):
                return torch.as_tensor(sigma)
            def __call__(self, x, sigma, labels):
                self.calls += 1
                return x * .5
        self.assertEqual(len(self.rows), 45)
        for row in self.rows[:9]:
            net = Net()
            torch.manual_seed(0)
            x = torch.randn(2, 3, 2, 2)
            if row['restart_k']:
                out = restart_sampler(net, x, num_steps=row['main_steps'],
                                      restart_info={'0': [3, row['restart_k'], .14, .30]})
            else:
                out = edm_sampler(net, x, num_steps=row['main_steps'])
            self.assertEqual(net.calls, row['nfe'])
            self.assertTrue(torch.isfinite(out).all())


class WatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []

    def evaluations(self, modes=('main18', 'nfe55')):
        result = []
        for mode in modes:
            rows = [dict(checkpoint=ck, condition=f'{ck}-k{k}', network=str(self.root / ck))
                    for ck in ('20k', '40k') for k in range(9)]
            completed = set()
            evaluation = SimpleNamespace(rows=rows, args=SimpleNamespace(exp_root=self.root),
                                         module=SimpleNamespace(aggregate=mock.Mock()),
                                         results=[], pause=mock.Mock())
            evaluation.completed = lambda rows, done=completed: all(r['condition'] in done for r in rows)
            def samples(rows, mode=mode):
                self.events.append((rows[0]['checkpoint'], mode, 'samples'))
            def evaluate(rows, mode=mode, done=completed, results=evaluation.results):
                self.events.append((rows[0]['checkpoint'], mode, 'fid'))
                done.update(r['condition'] for r in rows)
                results.extend(rows)
            evaluation.samples, evaluation.evaluate = samples, evaluate
            result.append(evaluation)
        return result

    def test_arrival_order_restart_complete_and_final_stop(self):
        evaluations = self.evaluations()
        now = [0]
        (self.root / '40k').write_bytes(b'model')
        probe = watch.SnapshotProbe(stable_sec=1, validate=lambda p: None, clock=lambda: now[0])
        def wait(seconds):
            now[0] += seconds
            if now[0] >= 2:
                (self.root / '20k').touch(exist_ok=True) if not (self.root / '20k').exists() else None
                if (self.root / '20k').stat().st_size == 0:
                    (self.root / '20k').write_bytes(b'model')
        watch.run_watch(evaluations, ['20k', '40k'], probe, 1, wait=wait)
        self.assertEqual(self.events[:4], [('40k', 'main18', 'samples'), ('40k', 'nfe55', 'samples'),
                                         ('40k', 'main18', 'fid'), ('40k', 'nfe55', 'fid')])
        self.assertEqual(len(self.events), 8)
        # Completed work must not wait, generate, or honor a late STOP.
        for evaluation in evaluations:
            evaluation.pause.check_exit.side_effect = watch.StopRequested()
        watch.run_watch(evaluations, ['20k', '40k'], probe, 1, wait=mock.Mock())
        self.assertEqual(len(self.events), 8)

    def test_both_available_use_ascending_order_and_single_mode(self):
        for ck in ('20k', '40k'):
            (self.root / ck).write_bytes(b'model')
        probe = watch.SnapshotProbe(stable_sec=0, validate=lambda p: None)
        for ck in ('20k', '40k'):
            probe.ready(self.root / ck)
        watch.run_watch(self.evaluations(('nfe55',)), ['20k', '40k'], probe, 1)
        self.assertEqual(self.events, [('20k', 'nfe55', 'samples'), ('20k', 'nfe55', 'fid'),
                                       ('40k', 'nfe55', 'samples'), ('40k', 'nfe55', 'fid')])

    def test_explicit_exit_during_checkpoint_wait(self):
        evaluations = self.evaluations()
        evaluations[0].pause.check_exit.side_effect = watch.StopRequested()
        with self.assertRaises(watch.StopRequested):
            watch.run_watch(evaluations, ['20k'], watch.SnapshotProbe(), 300)
        self.assertEqual(self.events, [])

    def test_all_90_conditions_complete(self):
        evaluations = self.evaluations()
        checkpoints = ['20k', '40k', '60k', '80k', '100k']
        for evaluation in evaluations:
            evaluation.rows.extend(dict(checkpoint=ck, condition=f'{ck}-k{k}',
                                        network=str(self.root / ck))
                                   for ck in checkpoints[2:] for k in range(9))
        probe = watch.SnapshotProbe(stable_sec=0, validate=lambda p: None)
        for ck in checkpoints:
            path = self.root / ck
            path.write_bytes(b'model')
            probe.ready(path)
        watch.run_watch(evaluations, checkpoints, probe, 300)
        self.assertEqual(sum(len(evaluation.results) for evaluation in evaluations), 90)
        self.assertEqual(len(self.events), 20)

    def test_checkpoint_arrival_with_pause_does_not_start_evaluation(self):
        evaluations = self.evaluations()
        (self.root / '20k').write_bytes(b'model')
        probe = watch.SnapshotProbe(stable_sec=0, validate=lambda p: None)
        probe.ready(self.root / '20k')
        class Paused(Exception):
            pass
        evaluations[0].samples = mock.Mock(side_effect=Paused('waiting at first boundary'))
        with self.assertRaises(Paused):
            watch.run_watch(evaluations, ['20k'], probe, 300)
        self.assertEqual(self.events, [])

    def test_lock_conflict_and_release(self):
        path = self.root / 'runs/main18/logs/.run.lock'
        with watch.lock(path):
            with self.assertRaisesRegex(ValueError, 'Another 8-15 runner'):
                with watch.lock(path):
                    self.fail('lock accepted duplicate')
        with watch.lock(path):
            pass

    def test_overall_stop_checked_by_mode_pause(self):
        with mock.patch.dict(os.environ, {'PAUSE_ENABLED': '0'}):
            pause = watch.ExperimentPause(self.root / 'runs/main18', '8-15-main18',
                                           self.root / 'summary', 'ref', overall_root=self.root)
            try:
                (self.root / 'STOP_AFTER_CURRENT_CONDITION').touch()
                with self.assertRaises(watch.StopRequested):
                    pause.check_exit()
            finally:
                pause.close('exit')

    def test_plan_and_help_without_models_gpu_or_writes(self):
        for mode, count in [('all', 90), ('main18', 45), ('nfe55', 45)]:
            command = [str(watch.ROOT / 'scripts/run.sh'), '--mode', mode,
                       '--exp-root', str(self.root / mode), '--plan-only']
            env = dict(os.environ, PYTHON=sys.executable, EDM_GPU_CONFIG='/missing')
            output = subprocess.check_output(command, env=env, text=True)
            self.assertIn(f'conditions={count}; FID-50k evaluations={count}', output)
            self.assertFalse((self.root / mode).exists())
        subprocess.check_output([str(watch.ROOT / 'scripts/run.sh'), '--help'],
                                env=env, text=True)


if __name__ == '__main__':
    unittest.main()
