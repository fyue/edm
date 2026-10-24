"""CPU checks for provenance, strict protocols, restart statistics and recovery."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('experiment04', Path(__file__).with_name('run_experiment.py'))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'logs').mkdir()
        self.args = SimpleNamespace(exp_root=self.root,
                                    fid_ref=experiment.REPO / 'fid-refs/cifar10-32x32.npz',
                                    gen_batch=64, fid_batch=64, torchrun='mock-torchrun')
        self.environment, self.originals = experiment.sources(self.args)
        self.rows = experiment.definitions()

    def imported(self, row):
        path, source = self.originals[row['condition']]
        return experiment.imported(self.args, row, path, source, self.environment)

    def generated(self, row, seed, fid):
        p = experiment.protocol(self.args, row, seed, self.environment)
        return dict(row, seeds=seed, num_images=50000, fid=fid, protocol=p,
                    protocol_id=experiment.util.digest(p), origin='generated', source_path='',
                    source_result=None, images=str(experiment.folder_for(self.args, row, seed)),
                    generate_sec=1.0, fid_sec=1.0)

    def test_actual_four_source_results_import_without_source_changes(self):
        before = {p: experiment.util.identity(p) for p, _ in self.originals.values()}
        results = [self.imported(r) for r in self.rows]
        experiment.validate(self.args, results, self.environment)
        self.assertEqual([r['fid'] for r in results], [1.80678, 1.79335, 1.8371, 1.81901])
        self.assertEqual(len(results), 4)
        self.assertEqual(before, {p: experiment.util.identity(p) for p in before})

    def test_provenance_rejects_changed_source_fid(self):
        result = self.imported(self.rows[0])
        result['source_result'] = dict(result['source_result'], fid=2.0)
        with self.assertRaisesRegex(ValueError, 'Imported source'):
            experiment.validate(self.args, [result], self.environment)

    def test_gpu_and_sampling_protocol_changes_rejected(self):
        for field, value in [('runtime', dict(self.environment, gpu_uuid='different-GPU')),
                             ('gen_batch', 32), ('network', {'sha256': '0'*64, 'size': 1})]:
            result = self.generated(self.rows[0], experiment.SEEDS[1], 1.8)
            result['protocol'][field] = value
            result['protocol_id'] = experiment.util.digest(result['protocol'])
            with self.assertRaisesRegex(ValueError, 'protocol'):
                experiment.validate(self.args, [result], self.environment)

    def test_duplicates_and_nonfinite_values_rejected(self):
        result = self.imported(self.rows[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            experiment.validate(self.args, [result, result], self.environment)
        for fid in [float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError):
                experiment.validate(self.args, [self.generated(self.rows[0], experiment.SEEDS[1], fid)], self.environment)

    def test_filters_reject_unknown_or_duplicate_values(self):
        self.assertEqual(experiment.select(experiment.CHECKPOINTS, '210k,200k'), ['200k', '210k'])
        for selector in ('220k', '200k,200k', ''):
            with self.assertRaises(ValueError):
                experiment.select(experiment.CHECKPOINTS, selector)

    def test_paired_statistics_and_training_delta(self):
        records = []
        for i, row in enumerate(self.rows):
            records.append(self.imported(row))
            for seed, fid in zip(experiment.SEEDS[1:], [1.8 + i*.01, 1.82 + i*.01]):
                records.append(self.generated(row, seed, fid))
        with mock.patch.object(experiment, 'plot'):
            experiment.aggregate(self.args, records, self.environment)
        stats = experiment.read(self.root / 'results/condition_stats.json')
        self.assertTrue(all(r['complete'] and r['n'] == 3 and r['std'] is not None for r in stats))
        paired = experiment.read(self.root / 'results/paired_improvements.json')
        first = next(r for r in paired if r['checkpoint'] == '200k' and r['seeds'] == experiment.SEEDS[0])
        self.assertAlmostEqual(first['improvement'], 0.01343)
        deltas = experiment.read(self.root / 'results/training_deltas.json')
        first = next(r for r in deltas if r['restart_k'] == 0 and r['seeds'] == experiment.SEEDS[0])
        self.assertAlmostEqual(first['delta_210k_minus_200k'], 0.03032)
        self.assertTrue(experiment.read(self.root / 'results/status.json')['complete'])

    def test_incomplete_statistics_do_not_claim_three_ranges(self):
        with mock.patch.object(experiment, 'plot'):
            experiment.aggregate(self.args, [self.imported(r) for r in self.rows], self.environment)
        stats = experiment.read(self.root / 'results/condition_stats.json')
        self.assertTrue(all(r['n'] == 1 and not r['complete'] and r['std'] is None for r in stats))
        self.assertFalse(experiment.read(self.root / 'results/status.json')['complete'])

    def test_resume_generates_only_missing_seed_and_validates_images(self):
        from PIL import Image
        row, seed = self.rows[0], experiment.SEEDS[1]
        folder = experiment.folder_for(self.args, row, seed)
        protocol = experiment.protocol(self.args, row, seed, self.environment)
        experiment.util.ensure_protocol(folder, protocol)
        # Small inventory, but use the real seed paths, PNG validation and missing-range logic.
        allowed = [50000, 50001]
        path = folder / experiment.util.image_relative(50000)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 32)).save(path)
        calls = []

        def run_logged(command, log):
            calls.append(command)
            if 'generate.py' in command:
                self.assertIn('--seeds=50001-50001', command)
                Image.new('RGB', (32, 32)).save(folder / experiment.util.image_relative(50001))
                return 1.0, None
            return 1.0, 1.75

        with mock.patch.object(experiment.util, 'parse_seeds', return_value=allowed), \
                mock.patch.object(experiment.util, 'run_logged', side_effect=run_logged):
            experiment.evaluate(self.args, row, seed, self.environment)
        self.assertEqual(len(calls), 2)
        marker = experiment.marker(self.args, row, seed)
        self.assertEqual(experiment.read(marker)['fid'], 1.75)
        # A wrong seed/path must be rejected instead of silently entering FID.
        Image.new('RGB', (32, 32)).save(folder / '999999.png')
        with self.assertRaises(ValueError):
            experiment.util.inventory(folder, allowed)

    def coordinator(self, stop_after_first=False):
        args = self.args
        args.plan_only = args.verify_only = args.aggregate_only = args.samples_only = False
        args.gpu_worker = None
        args.checkpoint_filter = args.seed_filter = 'all'
        launched = []

        def launch(options, row, seeds):
            launched.append((row['condition'], seeds))
            value = self.generated(row, seeds, 2.0)  # Deliberately worse: no screening cutoff.
            experiment.util.write_json(experiment.marker(args, row, seeds), value)
            if stop_after_first:
                (args.exp_root / 'STOP_AFTER_CURRENT_CONDITION').touch()

        with mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment, 'launch', side_effect=launch), \
                mock.patch.object(experiment, 'plot'), \
                mock.patch.object(experiment.util, 'inventory'):
            experiment.main()
        return launched

    def test_eight_jobs_continue_even_when_fid_is_worse_and_skip_on_rerun(self):
        self.assertEqual(len(self.coordinator()), 8)
        self.assertEqual(len(self.coordinator()), 0)

    def test_stop_after_saved_condition_and_resume(self):
        self.assertEqual(len(self.coordinator(stop_after_first=True)), 1)
        self.assertEqual(len(experiment.load_results(self.args, self.environment)), 5)
        (self.root / 'STOP_AFTER_CURRENT_CONDITION').unlink()
        self.assertEqual(len(self.coordinator()), 7)

    def test_gpu_runtime_mismatch_prevents_generation(self):
        args = self.args
        args.plan_only = args.verify_only = args.aggregate_only = args.samples_only = False
        args.gpu_worker = '200k-k0'
        args.checkpoint_filter = 'all'
        args.seed_filter = experiment.SEEDS[1]
        with mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment.base, 'runtime', return_value=dict(self.environment, driver='different')), \
                mock.patch.object(experiment, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'runtime differs'):
                experiment.main()
            evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
