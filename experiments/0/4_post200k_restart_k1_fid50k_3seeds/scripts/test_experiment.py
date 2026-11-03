"""CPU checks for provenance, strict protocols, restart statistics and recovery."""
import copy
import importlib.util

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

    def generated(self, row, seed, fid, environment=None):
        p = experiment.protocol(self.args, row, seed, environment or self.environment)
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

    def test_sampling_protocol_changes_rejected(self):
        for field, value in [('gen_batch', 32), ('network', {'sha256': '0'*64, 'size': 1})]:
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
        args.plan_only = args.verify_only = args.aggregate_only = args.samples_only = args.refresh_imports = False
        args.checkpoint_filter = args.seed_filter = 'all'
        launched = []
        controller = mock.Mock(info={})

        def check_exit():
            if (args.exp_root / 'STOP_AFTER_CURRENT_CONDITION').exists():
                raise experiment.StopRequested()

        controller.check_exit.side_effect = check_exit
        controller.boundary.side_effect = lambda *a: check_exit()

        def evaluate(options, row, seeds, environment, **kwargs):
            launched.append((row['condition'], seeds))
            value = self.generated(row, seeds, 2.0)  # Deliberately worse: no screening cutoff.
            experiment.util.write_json(experiment.marker(args, row, seeds), value)
            if stop_after_first:
                (args.exp_root / 'STOP_AFTER_CURRENT_CONDITION').touch()

        with mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment, 'ConditionPause', return_value=controller), \
                mock.patch.object(experiment.base, 'runtime', return_value=self.environment), \
                mock.patch.object(experiment, 'evaluate', side_effect=evaluate), \
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

    def test_incomplete_gpu_runtime_metadata_prevents_generation(self):
        args = self.args
        args.plan_only = args.verify_only = args.aggregate_only = args.samples_only = args.refresh_imports = False
        args.checkpoint_filter = 'all'
        args.seed_filter = experiment.SEEDS[1]
        controller = mock.Mock(info={})
        with mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment, 'ConditionPause', return_value=controller), \
                mock.patch.object(experiment.base, 'runtime', return_value={'driver': 'missing-runtime'}), \
                mock.patch.object(experiment, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'runtime metadata'):
                experiment.main()
            evaluate.assert_not_called()
            controller.close.assert_called_once_with('error')

    def pascal_runtime(self):
        return dict(self.environment, gpu_name='NVIDIA TITAN X (Pascal)', gpu_uuid='Pascal-UUID',
                    hostname='other-pc', physical_gpu='2', pci_bus_id='00000000:03:00.0',
                    driver='another-driver', cuda='11.8', torch='2.0.0+cu118')

    def test_pascal_ranges_are_accepted_and_actual_runtime_is_preserved(self):
        records = [self.imported(r) for r in self.rows]
        for row in self.rows:
            for seed in experiment.SEEDS[1:]:
                records.append(self.generated(row, seed, 1.8, self.pascal_runtime()))
        experiment.validate(self.args, records, self.environment)
        with mock.patch.object(experiment, 'plot'):
            experiment.aggregate(self.args, records, self.environment)
        status = experiment.read(self.root / 'results/status.json')
        self.assertTrue(status['complete'])
        self.assertTrue(status['mixed_environments'])
        table = experiment.util.read_tsv(self.root / 'results/restart_fid_summary.tsv')
        self.assertTrue(any(r['gpu_name'] == 'NVIDIA TITAN X (Pascal)' for r in table))

    def test_same_seed_restart_pair_must_use_same_runtime(self):
        a = self.generated(self.rows[0], experiment.SEEDS[1], 1.8)
        b = self.generated(self.rows[1], experiment.SEEDS[1], 1.7, self.pascal_runtime())
        with self.assertRaisesRegex(ValueError, 'same GPU/software runtime'):
            experiment.validate(self.args, [a, b], self.environment)
        experiment.validate(self.args, [a], self.environment)
        with self.assertRaisesRegex(ValueError, 'same GPU/software runtime'):
            experiment.check_pair_runtime([a], self.rows[1], experiment.SEEDS[1], self.pascal_runtime())
        # Host/device numbering can differ while the actual GPU and numerical environment match.
        equivalent = dict(self.environment, hostname='renamed-host', physical_gpu='3', pci_bus_id='new-bus')
        experiment.check_pair_runtime([a], self.rows[1], experiment.SEEDS[1], equivalent)

    def test_absolute_paths_from_other_pc_are_relocated_for_validation(self):
        imported = self.imported(self.rows[0])
        imported['network'] = '/other-pc/repo/' + str(Path(self.rows[0]['network']).relative_to(experiment.REPO))
        imported['source_path'] = '/other-pc/repo/' + str(Path(imported['source_path']).relative_to(experiment.REPO))
        experiment.validate(self.args, [imported], self.environment)
        generated = self.generated(self.rows[0], experiment.SEEDS[1], 1.8, self.pascal_runtime())
        relative = experiment.folder_for(self.args, self.rows[0], experiment.SEEDS[1]).relative_to(self.root)
        generated['images'] = '/other-pc/output/' + str(relative)
        generated['network'] = imported['network']
        expected_folder = self.root / relative
        with mock.patch.object(experiment.util, 'ensure_protocol') as ensure, \
                mock.patch.object(experiment.util, 'inventory') as inventory:
            experiment.validate(self.args, [generated], self.environment, images=True)
        self.assertEqual(ensure.call_args.args[0], expected_folder)
        self.assertEqual(inventory.call_args.args[0], expected_folder)
        generated['images'] = '../unexpected/image-directory'
        with self.assertRaises(ValueError):
            experiment.validate(self.args, [generated], self.environment)

    def test_merge_separate_pc_roots_and_reject_duplicate_or_mixed_pairs(self):
        roots = [self.root / 'quadro', self.root / 'pascal']
        for row in self.rows:
            experiment.util.write_json(experiment.marker(SimpleNamespace(exp_root=roots[0]), row, experiment.SEEDS[0]), self.imported(row))
            for seed in experiment.SEEDS[1:]:
                result = self.generated(row, seed, 1.8, self.pascal_runtime())
                experiment.util.write_json(experiment.marker(SimpleNamespace(exp_root=roots[1]), row, seed), result)
        records = experiment.load_results(self.args, self.environment, roots=roots)
        self.assertEqual(len(records), 12)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            experiment.load_results(self.args, self.environment, roots=roots + roots[:1])

    def test_coordinator_accepts_pascal_and_records_it_in_new_results(self):
        args = self.args
        args.plan_only = args.verify_only = args.aggregate_only = args.samples_only = args.refresh_imports = False
        args.checkpoint_filter = '200k'
        args.seed_filter = experiment.SEEDS[1]
        controller = mock.Mock(info={})
        actual = self.pascal_runtime()

        def evaluate(options, row, seed, environment, **kwargs):
            self.assertEqual(environment, actual)
            experiment.util.write_json(experiment.marker(args, row, seed), self.generated(row, seed, 1.7, environment))

        with mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment, 'ConditionPause', return_value=controller), \
                mock.patch.object(experiment.base, 'runtime', return_value=actual), \
                mock.patch.object(experiment, 'evaluate', side_effect=evaluate), \
                mock.patch.object(experiment, 'plot'):
            experiment.main()
        records = experiment.load_results(args, self.environment)
        self.assertTrue(all(r['protocol']['runtime'] == actual for r in records))
        self.assertEqual(len(records), 2)
        controller.event.assert_called()

    def old_imports(self):
        records = []
        for row in self.rows:
            result = self.imported(row)
            result['protocol'].pop('pause_controller')
            result['protocol'].pop('pause_eta')
            result['protocol']['evaluation_runner'] = {'sha256': 'a'*64, 'size': 1}
            result['protocol_id'] = experiment.util.digest(result['protocol'])
            experiment.util.write_json(experiment.marker(self.args, row, experiment.SEEDS[0]), result)
            records.append(result)
        return records

    def test_refresh_retains_values_and_sources_and_backs_up_old_protocols(self):
        old = self.old_imports()
        with mock.patch.object(experiment, 'plot'), mock.patch.object(experiment, 'ConditionPause') as controller:
            experiment.refresh_imports(self.args, self.environment, self.originals)
            controller.assert_not_called()
        new = experiment.load_results(self.args, self.environment)
        self.assertEqual(sorted((r['condition'], r['fid'], r['source_path']) for r in old),
                         sorted((r['condition'], r['fid'], r['source_path']) for r in new))
        for r in old:
            backup = self.root / 'results/import_backups' / r['protocol_id'] / experiment.marker(self.args, r, r['seeds']).name
            self.assertEqual(experiment.read(backup), r)
        with mock.patch.object(experiment, 'plot'):
            experiment.refresh_imports(self.args, self.environment, self.originals)
        self.assertEqual(len(list((self.root / 'results/import_backups').rglob('*.json'))), 4)

    def test_refresh_rejects_tampering_before_any_rewrite(self):
        old = self.old_imports()
        damaged = copy.deepcopy(old[-1])
        damaged['protocol']['gen_batch'] = 32
        damaged['protocol_id'] = experiment.util.digest(damaged['protocol'])
        experiment.util.write_json(experiment.marker(self.args, damaged, damaged['seeds']), damaged)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            experiment.refresh_imports(self.args, self.environment, self.originals)
        self.assertFalse((self.root / 'results/import_backups').exists())
        self.assertEqual(experiment.read(experiment.marker(self.args, old[0], old[0]['seeds'])), old[0])

    def test_refresh_rejects_generated_and_partial_results(self):
        self.old_imports()
        row, seed = self.rows[0], experiment.SEEDS[1]
        value = self.generated(row, seed, 2.0)
        path = experiment.marker(self.args, row, seed)
        experiment.util.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'only source-imported'):
            experiment.refresh_imports(self.args, self.environment, self.originals)
        path.unlink()
        folder = experiment.folder_for(self.args, row, seed)
        folder.mkdir(parents=True)
        (folder / '.protocol.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'partial image'):
            experiment.refresh_imports(self.args, self.environment, self.originals)

    def test_phase_notifications_for_fid_and_smoke(self):
        row, seed = self.rows[0], experiment.SEEDS[1]
        pause = mock.Mock()
        with mock.patch.object(experiment.util, 'inventory', return_value={}), \
                mock.patch.object(experiment.util, 'parse_seeds', return_value=[50000]), \
                mock.patch.object(experiment.util, 'run_logged', return_value=(1.0, 1.7)):
            experiment.evaluate(self.args, row, seed, self.environment, pause=pause)
        self.assertEqual([c.args[0] for c in pause.phase.call_args_list],
                         ['validate', 'generate', 'validate', 'fid', 'save'])
        pause.reset_mock()
        with mock.patch.object(experiment.util, 'inventory', return_value={}), \
                mock.patch.object(experiment.util, 'run_logged', return_value=(1.0, None)) as run:
            experiment.evaluate(self.args, row, '0-1', self.environment, smoke=True, pause=pause)
        self.assertEqual([c.args[0] for c in pause.phase.call_args_list],
                         ['validate', 'generate', 'validate', 'saved'])
        self.assertEqual(run.call_count, 1)

    def test_completed_and_import_only_selections_never_start_controller(self):
        self.coordinator()
        self.args.seed_filter = 'all'
        with mock.patch.object(experiment, 'arguments', return_value=self.args), \
                mock.patch.object(experiment, 'ConditionPause') as controller, \
                mock.patch.object(experiment.util, 'inventory'), \
                mock.patch.object(experiment, 'plot'):
            experiment.main()
            self.args.seed_filter = experiment.SEEDS[0]
            experiment.main()
            controller.assert_not_called()

    def test_smoke_coordinator_sets_two_image_eta_and_closes_monitor(self):
        args = self.args
        args.plan_only = args.verify_only = args.aggregate_only = args.refresh_imports = False
        args.samples_only = True
        args.seed_filter = 'all'
        args.checkpoint_filter = '200k'
        pause = mock.Mock(info={})
        with mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment, 'ConditionPause', return_value=pause), \
                mock.patch.object(experiment.base, 'runtime', return_value=self.environment), \
                mock.patch.object(experiment, 'evaluate') as evaluate:
            experiment.main()
        self.assertEqual(pause.info['num_images'], 2)
        self.assertEqual([c.args[1] for c in pause.boundary.call_args_list], ['0-1', '0-1'])
        self.assertEqual(evaluate.call_count, 2)
        pause.close.assert_called_once_with('complete')

    def test_real_controller_initial_pause_and_pause_after_saved_condition(self):
        # Exercise 0-4 with the actual controller/ETA monitor, replacing only GPU access.
        import condition_pause
        args = self.args
        args.plan_only = args.verify_only = args.aggregate_only = args.samples_only = args.refresh_imports = False
        args.checkpoint_filter = '200k'
        args.seed_filter = experiment.SEEDS[1]
        pause_file = self.root / 'PAUSE'
        pause_file.touch()
        controllers, errors, evaluated, wait_calls = [], [], [], []
        original_popen = subprocess.Popen

        class FreeGPU:
            returncode = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def poll(self): return 0
            def wait(self): return 0
            def terminate(self): pass

        def popen(command, **kwargs):
            if command[:2] == ['bash', 'scripts/wait_for_gpu_and_run.sh']:
                wait_calls.append(command)
                return FreeGPU()
            return original_popen(command, **kwargs)

        def controller(*values):
            value = condition_pause.ConditionPause(*values)
            controllers.append(value)
            return value

        def evaluate(options, row, seed, environment, **kwargs):
            evaluated.append(row['condition'])
            experiment.util.write_json(experiment.marker(args, row, seed), self.generated(row, seed, 1.7))
            pause_file.touch()

        def work():
            try:
                experiment.main()
            except BaseException as exc:
                errors.append(exc)

        def wait_for(predicate):
            deadline = time.monotonic() + 8
            while not predicate():
                if errors or time.monotonic() > deadline:
                    self.fail(f'Timeout or controller error: {errors}')
                time.sleep(.01)

        with mock.patch.dict(os.environ, PAUSE_ENABLED='1', PAUSE_FILE=str(pause_file),
                             PAUSE_POLL_SEC='.01', PAUSE_ETA_UPDATE_SEC='.02'), \
                mock.patch.object(experiment, 'arguments', return_value=args), \
                mock.patch.object(experiment, 'ConditionPause', side_effect=controller), \
                mock.patch.object(condition_pause.subprocess, 'Popen', side_effect=popen), \
                mock.patch.object(condition_pause.subprocess, 'check_output', return_value=json.dumps({})), \
                mock.patch.object(experiment.base, 'runtime', return_value=self.environment), \
                mock.patch.object(experiment, 'evaluate', side_effect=evaluate), \
                mock.patch.object(experiment, 'plot'):
            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            try:
                wait_for(lambda: bool(controllers) and controllers[0].state.exists()
                         and experiment.read(controllers[0].state)['phase'] == 'waiting')
                self.assertEqual(wait_calls, [])
                self.assertEqual(evaluated, [])
                pause_file.unlink()
                wait_for(lambda: len(evaluated) == 1 and experiment.read(controllers[0].state)['phase'] == 'waiting')
                self.assertTrue(experiment.marker(args, self.rows[0], experiment.SEEDS[1]).exists())
                pause_file.unlink()
                thread.join(8)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len(evaluated), 2)
                self.assertEqual(len(wait_calls), 2)  # Once per condition, no double wait.
                self.assertTrue(pause_file.exists())  # Last condition finishes without waiting.
                self.assertEqual(experiment.read(controllers[0].state)['phase'], 'complete')
                self.assertIsNotNone(controllers[0].monitor.poll())
            finally:
                if thread.is_alive():
                    (self.root / 'STOP_AFTER_CURRENT_CONDITION').touch()
                    thread.join(8)


if __name__ == '__main__':
    unittest.main()
