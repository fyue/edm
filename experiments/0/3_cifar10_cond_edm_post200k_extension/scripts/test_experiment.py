"""CPU tests for recovery, paired evaluation, and checkpoint-boundary stopping."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('extension', Path(__file__).with_name('run_experiment.py'))
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = SimpleNamespace(exp_root=Path(self.temporary.name), target_kimg=210000,
                                    torchrun='torchrun', fid_ref=Path(self.temporary.name) / 'ref.npz',
                                    gen_batch=64, fid_batch=64)
        self.base_options = extension.read(extension.BASE_OPTIONS)

    def checkpoint(self, kimg):
        directory = extension.run_dir(self.args)
        directory.mkdir(parents=True, exist_ok=True)
        for path in extension.pair(directory, kimg):
            path.write_bytes(b'test checkpoint')
        options = dict(self.base_options, total_kimg=210000)
        (directory / 'training_options.json').write_text(json.dumps(options))

    def test_recovery_schedule_and_future_extension(self):
        self.assertEqual(extension.checkpoints(200000, 210000), [202500, 205000, 207500, 210000])
        self.checkpoint(202500)
        self.checkpoint(205000)
        current, state = extension.latest_state(self.args)
        self.assertEqual(current, 205000)
        command = extension.training_command(self.args, current, state)
        self.assertIn('--duration=210', command)
        self.assertIn('--checkpoint-kimgs=207500,210000', command)
        self.args.target_kimg = 220000
        self.assertEqual(extension.checkpoints(210000, 220000), [212500, 215000, 217500, 220000])

    def test_missing_or_empty_pair_rejected(self):
        self.checkpoint(202500)
        extension.pair(extension.run_dir(self.args), 202500)[1].write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'):
            extension.latest_state(self.args)

    def test_training_configuration_mismatch_rejected(self):
        self.checkpoint(202500)
        options = copy.deepcopy(self.base_options)
        options['optimizer_kwargs']['lr'] = 0.0002
        (extension.run_dir(self.args) / 'training_options.json').write_text(json.dumps(options))
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            extension.latest_state(self.args)

    def test_no_silent_target_rewind(self):
        self.checkpoint(220000)
        with self.assertRaisesRegex(ValueError, 'exceeds target'):
            extension.latest_state(self.args)

    def test_screening_boundary(self):
        self.assertTrue(extension.screening_passes(1.80, 1.79))
        self.assertTrue(extension.screening_passes(1.80679, 1.79679))
        self.assertFalse(extension.screening_passes(1.80679, 1.79680))
        self.assertFalse(extension.screening_passes(1.80679, 1.81))

    def test_gpu_or_protocol_mismatch_rejected(self):
        row = extension.row_for(self.args, 210000, 0)
        path = extension.result_path(self.args, row, extension.SEEDS[0])
        p = {'runtime': {'gpu_uuid': 'gpu-A'}, 'batch': 64}
        result = dict(kimg=210000, restart_k=0, seeds=extension.SEEDS[0], num_images=50000,
                      fid=1.7, protocol=p, protocol_id=extension.fid_tools.digest(p))
        extension.fid_tools.write_json(path, result)
        with mock.patch.object(extension, 'protocol', return_value=p):
            extension.verify_result(self.args, path, {}, images=False)
        changed = dict(p, runtime={'gpu_uuid': 'gpu-B'})
        with mock.patch.object(extension, 'protocol', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'protocol'):
                extension.verify_result(self.args, path, {}, images=False)

    def test_paired_means_and_incomplete_counts(self):
        records = []
        for seeds, a, b in zip(extension.SEEDS, [1.8, 1.82, 1.79], [1.78, 1.81, 1.80]):
            records.extend([dict(kimg=200000, restart_k=0, seeds=seeds, fid=a),
                            dict(kimg=210000, restart_k=0, seeds=seeds, fid=b)])
        extension.aggregate(self.args, records[:2])
        row = extension.read(self.args.exp_root / 'results/comparison.json')[0]
        self.assertEqual(row['n'], 1)
        self.assertIsNone(row['candidate_std'])
        self.assertFalse(row['complete'])
        extension.aggregate(self.args, records)
        row = extension.read(self.args.exp_root / 'results/comparison.json')[0]
        self.assertEqual(row['positive_ranges'], 2)
        self.assertEqual(row['assessment'], 'improvement_tendency')

    def coordinate(self, improvement, stop_training=False):
        args = self.args
        args.plan_only = args.verify_only = args.eval_only = args.train_only = False
        args.worker = None
        args.fid_stage = 'auto'
        calls = []

        def launch(options, job, cp=200000, k=0, seeds=extension.SEEDS[0]):
            calls.append((job, cp, k, seeds))
            if job == 'train':
                self.checkpoint(202500 if stop_training else 210000)
                if stop_training:
                    (args.exp_root / 'STOP_AFTER_CURRENT_STAGE').touch()
                return
            row = extension.row_for(args, cp, k)
            path = extension.result_path(args, row, seeds)
            extension.fid_tools.write_json(path, dict(kimg=cp, restart_k=k, seeds=seeds,
                                           fid=1.8 if cp == 200000 else 1.8 - improvement))
            extension.fid_tools.write_json(args.exp_root / 'results/runtime.json', {})

        with mock.patch.object(extension, 'arguments', return_value=args), \
                mock.patch.object(extension, 'ensure_manifest'), \
                mock.patch.object(extension, 'health'), \
                mock.patch.object(extension, 'verify_result', side_effect=lambda a, p, e, images=True: extension.read(p)), \
                mock.patch.object(extension, 'launch', side_effect=launch):
            extension.main()
        return calls

    def test_auto_confirmation_runs_all_twelve_evaluations(self):
        calls = self.coordinate(0.02)
        self.assertEqual(sum(call[0] == 'eval' for call in calls), 12)
        summary = extension.read(self.args.exp_root / 'results/comparison.json')
        self.assertTrue(all(r['n'] == 3 for r in summary))

    def test_auto_non_improvement_stops_after_four_evaluations(self):
        calls = self.coordinate(-0.01)
        self.assertEqual(sum(call[0] == 'eval' for call in calls), 4)
        self.assertFalse(extension.read(self.args.exp_root / 'results/screening-210000.json')['passed'])

    def test_training_stop_skips_fid(self):
        calls = self.coordinate(0.02, stop_training=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(extension.latest_state(self.args)[0], 202500)


import torch
from training import training_loop


class TinyDataset(torch.utils.data.Dataset):
    resolution = num_channels = label_dim = 1

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.zeros(1, 1, 1, dtype=torch.uint8), torch.ones(1)


class TinyNet(torch.nn.Module):
    img_resolution = img_channels = label_dim = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((1,), 0.5))


class TinyLoss:
    def __call__(self, net, images, labels, augment_pipe):
        return (net.weight - labels).square()


class TrainingStopTests(unittest.TestCase):
    def test_real_cpu_loop_saves_pair_before_stopping_then_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stop = directory / 'STOP'
            stop.touch()

            def construct(class_name, **kwargs):
                return {'dataset': lambda: TinyDataset(), 'network': lambda: TinyNet(),
                        'loss': lambda: TinyLoss(), 'optimizer': lambda: torch.optim.Adam(**kwargs)}[class_name]()

            with mock.patch.object(training_loop.dnnlib.util, 'construct_class_by_name', side_effect=construct), \
                    mock.patch.object(torch.nn.parallel, 'DistributedDataParallel', side_effect=lambda net, **kw: net), \
                    mock.patch.object(training_loop.misc, 'print_module_summary'), \
                    mock.patch.object(training_loop.misc, 'check_ddp_consistency'), \
                    mock.patch.object(torch.distributed, 'barrier'), \
                    mock.patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                    mock.patch.object(torch.cuda, 'max_memory_reserved', return_value=0), \
                    mock.patch.object(torch.cuda, 'reset_peak_memory_stats'):
                options = dict(run_dir=str(directory), dataset_kwargs={'class_name': 'dataset'},
                               network_kwargs={'class_name': 'network'}, loss_kwargs={'class_name': 'loss'},
                               optimizer_kwargs={'class_name': 'optimizer', 'lr': 0.001},
                               batch_size=512, batch_gpu=512, total_kimg=3, checkpoint_kimgs=[1, 2, 3],
                               snapshot_ticks=None, state_dump_ticks=None, device=torch.device('cpu'),
                               stop_file=str(stop))
                training_loop.training_loop(**options)
                self.assertTrue(all(p.exists() for p in extension.pair(directory, 1)))
                self.assertFalse(extension.pair(directory, 2)[0].exists())
                first = torch.load(extension.pair(directory, 1)[1], map_location='cpu')
                self.assertEqual(next(iter(first['optimizer_state']['state'].values()))['step'].item(), 2)
                stop.unlink()
                options.update(resume_pkl=str(extension.pair(directory, 1)[0]),
                               resume_state_dump=str(extension.pair(directory, 1)[1]), resume_kimg=1)
                training_loop.training_loop(**options)
                self.assertTrue(all(p.exists() for p in extension.pair(directory, 3)))
                saved = torch.load(extension.pair(directory, 3)[1], map_location='cpu')
                self.assertTrue(saved['optimizer_state']['state'])
                self.assertEqual(next(iter(saved['optimizer_state']['state'].values()))['step'].item(), 6)


if __name__ == '__main__':
    unittest.main()
