"""CPU tests for launch durability and real checkpoint validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_experiment as runner


class LaunchTests(unittest.TestCase):
    def test_fresh_seeds_include_failed_and_unfinished_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'history.jsonl'
            self.assertEqual(runner.choose_seed(200000, []), 200000)
            runner.append_event(ledger, dict(event='start', seed=200000))
            self.assertEqual(runner.choose_seed(200000, runner.read_history(ledger)), 200001)
            runner.append_event(ledger, dict(event='end', seed=200000, status='failed'))
            runner.append_event(ledger, dict(event='start', seed=200001))
            self.assertEqual(runner.choose_seed(200000, runner.read_history(ledger)), 200002)
            self.assertEqual(runner.choose_seed(210000, runner.read_history(ledger)), 210000)
            with ledger.open('a') as stream:
                stream.write('{broken')
            with self.assertRaises(json.JSONDecodeError):
                runner.read_history(ledger)

    def make_pair(self, directory, kimg):
        import pickle
        import torch
        net = torch.nn.Linear(2, 2)
        opt = torch.optim.Adam(net.parameters())
        net(torch.ones(1, 2)).sum().backward()
        opt.step()
        snapshot, state = runner.pair(directory, kimg)
        directory.mkdir(parents=True, exist_ok=True)
        with snapshot.open('wb') as stream:
            pickle.dump(dict(ema=net), stream)
        torch.save(dict(net=net, optimizer_state=opt.state_dict()), state)

    def test_latest_valid_pair_skips_incomplete_and_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pair(root, 210000)
            runner.pair(root, 220000)[1].write_bytes(b'broken')
            for path in runner.pair(root, 230000):
                path.write_bytes(b'broken')
            start, state, rejected = runner.select_resume(root)
            self.assertEqual(start, 210000)
            self.assertEqual(state, runner.pair(root, 210000)[1])
            self.assertEqual([r['kimg'] for r in rejected], [230000, 220000])

    def test_base_fallback_and_nonfinite_refusal(self):
        import pickle
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pair(root / 'base', 200000)
            self.make_pair(root / 'run', 210000)
            net = torch.nn.Linear(2, 2)
            with torch.no_grad():
                net.weight.fill_(float('nan'))
            with runner.pair(root / 'run', 210000)[0].open('wb') as stream:
                pickle.dump(dict(ema=net), stream)
            start, _, rejected = runner.select_resume(root / 'run', base=root / 'base')
            self.assertEqual(start, 200000)
            self.assertIn('Nonfinite', rejected[0]['reason'])
            runner.pair(root / 'base', 200000)[1].write_bytes(b'')
            with self.assertRaises(ValueError):
                runner.select_resume(root / 'run', base=root / 'base')

    def test_plan_and_dry_run_do_not_reserve_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ('--plan-only', '--dry-run'):
                with mock.patch('sys.argv', ['runner', mode, '--exp-root', str(root)]), \
                     mock.patch.object(runner, 'validate_base'), \
                     mock.patch.object(runner, 'select_resume', return_value=(200000, Path('/base/training-state-200000.pt'), [])), \
                     mock.patch.object(runner.subprocess, 'run'):
                    runner.main()
            self.assertFalse((root / 'results/launch_history.jsonl').exists())

    def test_real_failed_launch_consumes_seed(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'training-runs/run'
            gpu_env = dict(GPU_INDEX='0', CUDA_VISIBLE_DEVICES='0',
                           CUDA_DEVICE_ORDER='PCI_BUS_ID', EXPECTED_GPU_NAME='test GPU')
            with mock.patch.dict(os.environ, gpu_env), \
                 mock.patch('torch.cuda.device_count', return_value=1), \
                 mock.patch.object(runner.subprocess, 'check_output', return_value='test GPU'), \
                 mock.patch.object(runner, 'ensure_manifest'), \
                 mock.patch.object(runner, 'select_resume', return_value=(200000, Path('/base/training-state-200000.pt'), [])), \
                 mock.patch.object(runner.subprocess, 'call', return_value=7):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, 'status 7'):
                        runner.execute(root, run_dir)
            history = runner.read_history(root / 'results/launch_history.jsonl')
            self.assertEqual([r['seed'] for r in history if r['event'] == 'start'], [200000, 200001])
            self.assertTrue(all(r['status'] == 'failed' for r in history if r['event'] == 'end'))

    def test_completion_requires_every_pair_progress_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'run'
            with mock.patch.object(runner, 'validate_pair', side_effect=lambda d, k: dict(
                    kimg=k, snapshot='snapshot', snapshot_bytes=1, state='state', state_bytes=1)), \
                 mock.patch.object(runner, 'read_progress', return_value=(300000.128, 0.1)):
                self.assertFalse(runner.verify(root, run_dir))
                runner.append_event(root / 'results/launch_history.jsonl', dict(event='start', seed=200000))
                self.assertTrue(runner.verify(root, run_dir))
                with mock.patch.object(runner, 'validate_pair', side_effect=ValueError('bad pair')):
                    self.assertFalse(runner.verify(root, run_dir))
                self.assertFalse(json.loads((root / 'results/health.json').read_text())['complete'])

    def test_training_command(self):
        args = runner.command(Path('/out'), Path('/base/training-state-200000.pt'), 200000, 200000)
        self.assertIn('--duration=300', args)
        self.assertIn('--seed=200000', args)
        self.assertIn('--snap=0', args)
        self.assertIn('--dump=0', args)
        targets = next(a.split('=', 1)[1] for a in args if a.startswith('--checkpoint-kimgs='))
        self.assertEqual(list(map(int, targets.split(','))), runner.TARGETS)


if __name__ == '__main__':
    unittest.main()
