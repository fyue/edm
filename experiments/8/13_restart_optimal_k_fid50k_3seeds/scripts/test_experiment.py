#!/usr/bin/env python3
"""CPU tests for imported results, paired statistics, resume and pause lifecycle."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import run_experiment as experiment
from condition_pause import ConditionPause, StopRequested
import pause_eta


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = experiment.conditions()
        self.args = SimpleNamespace(exp_root=self.root, fid_ref=self.root / 'ref.npz', gen_batch=64,
                                    fid_batch=64, torchrun='mock-torchrun')
        self.identity = {'sha256': 'a' * 64, 'size': 1}
        patcher = mock.patch.object(experiment.old, 'identity', return_value=self.identity)
        patcher.start()
        self.addCleanup(patcher.stop)

    def result(self, row, seed, fid):
        p = experiment.protocol(row, seed, self.args)
        return dict(row, seeds=seed, num_images=50000, fid=fid,
                    protocol=p, protocol_id=experiment.old.digest(p), origin='generated',
                    fid_ref='ref.npz', fid_sec=1, images=str(self.root / 'images'),
                    generated_images=50000, generate_sec=2, gpu_name='Mock GPU')

    def test_fixed_conditions_and_selection(self):
        self.assertEqual(len(self.rows), 12)
        self.assertEqual([(r['checkpoint'], r['restart_k']) for r in self.rows],
                         [(cp, k) for cp, best in experiment.OPTIMAL.items() for k in (0, best)])
        self.assertEqual(experiment.select(experiment.OPTIMAL, '20k,180k'), ['20k', '180k'])
        with self.assertRaises(ValueError):
            experiment.select(experiment.SEEDS, '0-50000')

    def test_stats_and_paired_differences_missing_and_duplicate(self):
        a, b = self.rows[:2]
        results = [self.result(r, seed, value) for r, values in [(a, [3, 4]), (b, [2, 2])]
                   for seed, value in zip(experiment.SEEDS[1:], values)]
        with mock.patch.object(experiment, 'plot'):
            experiment.aggregate(self.root, results, self.rows)
        paired = json.loads((self.root / 'results/paired_improvements.json').read_text())
        self.assertEqual([r['improvement'] for r in paired], [1, 2])
        stats = json.loads((self.root / 'results/improvement_stats.json').read_text())[0]
        self.assertEqual(stats['mean'], 1.5)
        self.assertAlmostEqual(stats['std'], 2 ** -.5)
        self.assertFalse(stats['complete'])
        self.assertEqual(stats['positive_ranges'], 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            experiment.validate(results + results[:1], {r['condition']: r for r in self.rows})
        self.assertEqual(experiment.stats([1, 2, 3]), dict(n=3, mean=2, std=1, complete=True))
        self.assertIsNone(experiment.stats([])['std'])

    def test_merge_and_protocol_mismatch(self):
        results = [self.result(self.rows[0], seed, 2) for seed in experiment.SEEDS[1:]]
        roots = [self.root / 'pc-a', self.root / 'pc-b']
        for root, row in zip(roots, results):
            experiment.old.write_json(experiment.marker(root, row), row)
        self.assertEqual(len(experiment.load_results(roots, self.rows)), 2)
        bad = copy.deepcopy(results[1])
        bad['protocol']['fid_ref']['sha256'] = 'b' * 64
        bad['protocol_id'] = experiment.old.digest(bad['protocol'])
        with self.assertRaisesRegex(ValueError, 'Mixed code'):
            experiment.validate([results[0], bad], {r['condition']: r for r in self.rows})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            experiment.load_results(roots + roots[:1], self.rows)

    def test_actual_legacy_sources_and_reuse_reject_changed_code(self):
        sources = experiment.legacy_results(self.rows)
        self.assertEqual(len(sources), 12)
        row = self.rows[0]
        source, path = sources[row['condition']]
        p = copy.deepcopy(source['protocol'])
        p.update(experiment='8-13', base_runner=p['runner'])
        with mock.patch.object(experiment, 'protocol', return_value=p):
            result = experiment.imported(row, source, path, self.args)
            experiment.validate([result], {r['condition']: r for r in self.rows})
            self.assertEqual(result['fid'], source['fid'])
            self.assertEqual(result['source_result'], source)
            p['generate'] = self.identity
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                experiment.imported(row, source, path, self.args)

    def test_resume_only_generates_missing_seeds_before_fid(self):
        row = self.rows[1]
        pause = mock.Mock()
        present = {50000}
        commands = []
        def inventory(folder, allowed, complete=False):
            if complete:
                self.assertEqual(present, {50000, 50001, 50002})
            return {seed: folder / str(seed) for seed in present}
        def run(command, log):
            commands.append(command)
            if 'generate.py' in command:
                self.assertIn('--seeds=50001-50002', command)
                present.update([50001, 50002])
                return 1, None
            return 2, 1.25
        with mock.patch.object(experiment.old, 'parse_seeds', return_value=[50000, 50001, 50002]), \
                mock.patch.object(experiment.old, 'inventory', side_effect=inventory), \
                mock.patch.object(experiment.old, 'run_logged', side_effect=run), \
                mock.patch.dict(os.environ, EXPECTED_GPU_NAME='Mock GPU'):
            result = experiment.evaluate(row, experiment.SEEDS[1], self.args, pause)
        self.assertEqual(result['generated_images'], 2)
        self.assertEqual(result['fid'], 1.25)
        self.assertIn('fid.py', commands[-1])
        self.assertEqual([c.args[0] for c in pause.phase.call_args_list],
                         ['validate', 'generate', 'validate', 'fid', 'save'])

    def test_main_saves_before_pause_skips_completed_and_finishes_last_job(self):
        self.args.__dict__.update(plan_only=False, verify_only=False, aggregate_only=False,
                                  samples_only=False, result_roots=None, checkpoint_filter='20k', seed_filter='all')
        legacy = experiment.legacy_results(self.rows[:2])
        def reused(row, source, path, args):
            r = self.result(row, experiment.SEEDS[0], source['fid'])
            r.update(origin='8-11', source_result=source, source_path=str(path))
            return r
        for trigger in ('generate', 'fid'):
            with self.subTest(trigger=trigger), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.args.exp_root = root
                events, controllers, errors = [], [], []
                pause_file = root / 'PAUSE'
                # Keep the actual parent, ETA observer and boundary logic; fake only GPU commands.
                def controller(*args):
                    value = ConditionPause(*args)
                    controllers.append(value)
                    return value
                def evaluate(row, seed, args, control):
                    events.append((row['condition'], seed))
                    control.phase(trigger, row, seed)
                    pause_file.touch()
                    return self.result(row, seed, 2)
                def output(command, **kwargs):
                    return json.dumps(dict(GPU_INDEX='7', EXPECTED_GPU_NAME='Mock GPU',
                                           CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='7'))
                real_popen = subprocess.Popen
                def launch(command, **kwargs):
                    if command[:2] == ['bash', 'scripts/wait_for_gpu_and_run.sh']:
                        return real_popen(['/bin/true'])
                    return real_popen(command, **kwargs)
                def run():
                    try:
                        experiment.main()
                    except BaseException as exc:
                        errors.append(exc)
                env = dict(PAUSE_FILE=str(pause_file), PAUSE_ENABLED='1', PAUSE_POLL_SEC='0.01', PAUSE_ETA_UPDATE_SEC='0.02')
                with mock.patch.dict(os.environ, env), \
                        mock.patch.object(experiment, 'arguments', return_value=self.args), \
                        mock.patch.object(experiment, 'imported', side_effect=reused), \
                        mock.patch.object(experiment, 'legacy_results', return_value=legacy), \
                        mock.patch.object(experiment, 'aggregate'), \
                        mock.patch.object(experiment, 'evaluate', side_effect=evaluate), \
                        mock.patch.object(experiment, 'ConditionPause', side_effect=controller), \
                        mock.patch('condition_pause.subprocess.Popen', side_effect=launch), \
                        mock.patch('condition_pause.subprocess.check_output', side_effect=output):
                    thread = threading.Thread(target=run, daemon=True)
                    thread.start()
                    try:
                        for expected in (1, 2, 3):
                            deadline = time.monotonic() + 8
                            while True:
                                state = pause_eta.read_json(controllers[0].state) if controllers else {}
                                if len(events) == expected and state.get('phase') == 'waiting':
                                    break
                                if errors or time.monotonic() > deadline:
                                    self.fail(f'Pause did not follow saved result: {errors}, {state}')
                                time.sleep(.01)
                            # Two imports plus every completed new evaluation are already durable.
                            self.assertEqual(len(list((root / 'results').glob('*.json'))), 2 + expected)
                            pause_file.unlink()
                        thread.join(8)
                        self.assertFalse(thread.is_alive())
                        self.assertEqual(errors, [])
                        self.assertEqual(len(events), 4)
                        self.assertTrue(pause_file.exists())  # The final result never waits.
                        self.assertEqual(pause_eta.read_json(controllers[0].state)['phase'], 'complete')
                        # Re-running an already completed selection never starts a GPU controller.
                        stored = [self.result(r, seed, 2) for r in self.rows[:2] for seed in experiment.SEEDS]
                        with mock.patch.object(experiment, 'load_results', return_value=stored):
                            experiment.main()
                        self.assertEqual(len(controllers), 1)
                    finally:
                        if thread.is_alive():
                            (root / 'STOP_AFTER_CURRENT_CONDITION').touch()
                            thread.join(8)


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pause_file = self.root / 'PAUSE'
        self.row = dict(condition='180k-k1', checkpoint='180k', restart_k=1)
        bindir = self.root / 'bin'
        bindir.mkdir()
        gpu = bindir / 'nvidia-smi'
        gpu.write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
with (root/'events').open('a') as f: f.write('gpu ' + ' '.join(sys.argv[1:]) + '\\n')
if '--query-gpu=name' in sys.argv:
 print('Mock GPU')
else:
 if (root/'repause').exists():
  (root/'repause').unlink(); (root/'PAUSE').touch()
 if (root/'busy').exists(): print('1234, job, 1')
''')
        gpu.chmod(0o755)
        config = self.root / 'gpu.env'
        config.write_text('GPU_INDEX=7\nEXPECTED_GPU_NAME="Mock GPU"\n')
        env = dict(TEST_ROOT=str(self.root), PATH=f'{bindir}:{os.environ["PATH"]}',
                   EDM_GPU_CONFIG=str(config), PAUSE_FILE=str(self.pause_file),
                   PAUSE_ENABLED='1', PAUSE_POLL_SEC='0.02', PAUSE_ETA_UPDATE_SEC='0.03',
                   CHECK_INTERVAL_SEC='1', GPU_INDEX='99', EXPECTED_GPU_NAME='Wrong GPU')
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.controller = ConditionPause(self.root, '8-13', self.root / 'summary.tsv', 'ref.npz')
        self.addCleanup(self.controller.close)
        self.errors = []

    def wait_for(self, predicate):
        end = time.monotonic() + 8
        while not predicate():
            if time.monotonic() > end:
                self.fail('Timed out; logs=' + (self.root / 'logs/pause.log').read_text())
            time.sleep(.01)

    def start_boundary(self):
        def work():
            try:
                self.controller.boundary(self.row, experiment.SEEDS[1])
            except BaseException as exc:
                self.errors.append(exc)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.addCleanup(lambda: self.controller.stop_file.touch() if thread.is_alive() else None)
        return thread

    def text(self):
        path = self.root / 'logs/pause.log'
        return path.read_text() if path.exists() else ''

    def test_initial_pause_busy_gpu_repause_and_repeat(self):
        self.pause_file.touch()
        (self.root / 'busy').touch()
        thread = self.start_boundary()
        self.wait_for(lambda: 'PAUSE before' in self.text())
        self.assertFalse((self.root / 'events').exists())
        self.pause_file.unlink()
        self.wait_for(lambda: (self.root / 'logs/schedule-after-gpu-free.log').exists()
                      and 'still busy' in (self.root / 'logs/schedule-after-gpu-free.log').read_text())
        (self.root / 'repause').touch()
        (self.root / 'busy').unlink()
        self.wait_for(lambda: self.text().count('PAUSE before') == 2)
        self.pause_file.unlink()
        thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')
        self.assertIn('-i 7', (self.root / 'events').read_text())
        self.pause_file.touch()
        thread = self.start_boundary()
        self.wait_for(lambda: self.text().count('PAUSE before') == 3)
        self.pause_file.unlink()
        thread.join(8)
        self.assertFalse(thread.is_alive())

    def test_exit_takes_priority_and_waiter_can_exit(self):
        self.pause_file.touch()
        self.controller.stop_file.touch()
        with self.assertRaises(StopRequested):
            self.controller.boundary(self.row, experiment.SEEDS[1])
        self.assertFalse((self.root / 'events').exists())
        self.controller.stop_file.unlink()
        self.pause_file.unlink()
        (self.root / 'busy').touch()
        thread = self.start_boundary()
        self.wait_for(lambda: (self.root / 'events').exists())
        self.controller.stop_file.touch()
        thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(self.errors[-1], StopRequested)

    def test_phase_report_contains_experiment_seed_and_preserves_deleted_stop(self):
        self.controller.phase('fid', self.row, experiment.SEEDS[1])
        state = pause_eta.read_json(self.controller.state)
        report = pause_eta.report(state, self.root / 'summary.tsv', self.controller.history, os.getpid())
        self.assertIn('experiment=8-13', report)
        self.assertIn('seeds=50000-99999', report)
        self.assertIn('FID計算中', report)
        self.assertIn('推定中', report)
        self.assertFalse(pause_eta.append_locked(self.pause_file, report, existing_only=True))
        self.assertFalse(self.pause_file.exists())


if __name__ == '__main__':
    unittest.main()
