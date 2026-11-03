"""CPU-only checks for explicit, immutable legacy runner compatibility."""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from PIL import Image

import run_experiment as experiment
import runner_compatibility as compatibility
import test_experiment as fixtures


class CompatibilityTests(unittest.TestCase):
    setUp = fixtures.ExperimentTests.setUp

    def prepare(self):
        self.args.config = experiment.DEFAULT_ROOT / 'config' / experiment.CONFIG_NAME
        self.args.checkpoint_filter = '20k'
        self.args.condition_filter = self.row['condition']
        self.args.fid_stage = '25000'
        self.args.samples_only = False
        (self.root / 'logs').mkdir()
        self.legacy_source = b'fixture legacy runner'
        self.legacy = compatibility.content_identity(self.legacy_source)
        for patch in (mock.patch.object(compatibility, 'LEGACY', self.legacy),
                      mock.patch.object(compatibility, 'reviewed_source', return_value=self.legacy_source),
                      mock.patch.object(experiment, 'load_conditions', return_value=self.rows)):
            patch.start()
            self.addCleanup(patch.stop)
        self.value = dict(experiment.protocol(self.row, self.args), runner=dict(self.legacy))
        self.pool = self.root / 'images' / self.row['condition'] / 'pool'
        experiment.ensure_protocol(self.pool, self.value)
        target = self.pool / experiment.image_relative(0)
        target.parent.mkdir(parents=True)
        Image.new('RGB', (32, 32)).save(target)

    def test_registration_idempotence_and_preservation(self):
        self.prepare()
        before = (self.pool / '.protocol.json').read_bytes()
        compatibility.register(self.args, experiment)
        path = self.root / compatibility.DIRECTORY / 'registration.json'
        registration = path.read_bytes()
        compatibility.register(self.args, experiment)
        self.assertEqual(path.read_bytes(), registration)
        self.assertEqual((self.pool / '.protocol.json').read_bytes(), before)
        self.assertEqual(experiment.protocol(self.row, self.args), self.value)
        other = copy.copy(self.args)
        other.exp_root = self.root / 'other'
        self.assertNotEqual(experiment.protocol(self.row, other)['runner'], self.legacy)

    def test_unregistered_and_unknown_legacy_are_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
            experiment.ensure_protocol(self.pool, experiment.protocol(self.row, self.args))
        changed = dict(self.value, runner={'sha256': '0' * 64, 'size': 1})
        experiment.write_json(self.pool / '.protocol.json', changed)
        with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
            compatibility.register(self.args, experiment)
        self.assertFalse((self.root / compatibility.DIRECTORY / 'registration.json').exists())

    def test_registered_runner_change_and_reregistration_are_rejected(self):
        self.prepare()
        compatibility.register(self.args, experiment)
        sources = compatibility.files(experiment)
        changed = self.root / 'changed-runner.py'
        changed.write_bytes(sources['runner'].read_bytes() + b'\n# changed\n')
        with mock.patch.object(compatibility, 'files', return_value=dict(sources, runner=changed)):
            with self.assertRaisesRegex(ValueError, 'Registered code changed: runner'):
                experiment.protocol(self.row, self.args)
            with self.assertRaisesRegex(ValueError, 'Registered code changed: runner'):
                compatibility.register(self.args, experiment)

    def test_helper_and_snapshot_changes_are_rejected(self):
        self.prepare()
        compatibility.register(self.args, experiment)
        path = self.root / compatibility.DIRECTORY / 'compatibility.source'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            experiment.protocol(self.row, self.args)

    def test_parameters_model_reference_and_generator_changes_are_rejected(self):
        self.prepare()
        compatibility.register(self.args, experiment)
        self.args.gen_batch = 32
        with self.assertRaisesRegex(ValueError, 'parameters'):
            experiment.protocol(self.row, self.args)
        self.args.gen_batch = 64
        original_identity = experiment.identity
        for filename in ('generate.py', 'fid.py'):
            def identity(path, filename=filename):
                value = original_identity(path)
                return dict(value, sha256='0' * 64) if Path(path).name == filename else value
            with mock.patch.object(experiment, 'identity', side_effect=identity):
                with self.assertRaisesRegex(ValueError, 'parameters'):
                    experiment.protocol(self.row, self.args)
        reference = self.args.fid_ref.read_bytes()
        self.args.fid_ref.write_bytes(b'changed reference')
        with self.assertRaisesRegex(ValueError, 'parameters'):
            experiment.protocol(self.row, self.args)
        self.args.fid_ref.write_bytes(reference)
        experiment.IDENTITIES.clear()
        Path(self.row['network']).write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'model changed'):
            experiment.protocol(self.row, self.args)

    def test_progress_and_image_corruption_prevent_registration(self):
        self.prepare()
        progress = self.root / 'progress' / f"{self.row['condition']}-fid25000.json"
        experiment.write_json(progress, {'protocol_id': 'invalid'})
        with self.assertRaisesRegex(ValueError, 'Progress protocol mismatch'):
            compatibility.register(self.args, experiment)
        progress.unlink()
        (self.pool / experiment.image_relative(0)).write_bytes(b'invalid PNG')
        with self.assertRaises(OSError):
            compatibility.register(self.args, experiment)

    def test_registration_is_locked_and_atomic(self):
        import fcntl
        self.prepare()
        with (self.root / 'logs/.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                compatibility.register(self.args, experiment)
        with mock.patch.object(experiment, 'write_json', side_effect=OSError('save failed')):
            with self.assertRaisesRegex(OSError, 'save failed'):
                compatibility.register(self.args, experiment)
        self.assertFalse((self.root / compatibility.DIRECTORY / 'registration.json').exists())
        compatibility.register(self.args, experiment)

    def test_execution_provenance_records_actual_code(self):
        self.prepare()
        compatibility.register(self.args, experiment)
        compatibility.record_execution(self.args, experiment)
        paths = list((self.root / 'logs/runner-executions').glob('*.json'))
        self.assertEqual(len(paths), 1)
        value = json.loads(paths[0].read_text())
        self.assertEqual(value['compatibility_basis'], self.legacy)
        self.assertEqual(value['actual_files']['runner'], experiment.identity(Path(experiment.__file__)))
        self.assertNotEqual(value['actual_files']['runner'], self.legacy)

    def test_modified_sampling_conditions_are_still_rejected(self):
        self.prepare()
        compatibility.register(self.args, experiment)
        changed = dict(self.row, t_min=0.15)
        with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
            experiment.ensure_protocol(self.pool, experiment.protocol(changed, self.args))

    def test_source_review_rejects_nonmatching_generation_function(self):
        source = b'def generate_command():\n    return ["different"]\n'
        with mock.patch.object(compatibility, 'LEGACY', compatibility.content_identity(source)), \
             mock.patch.object(compatibility.subprocess, 'check_output', return_value=source):
            with self.assertRaisesRegex(ValueError, 'Sampling/validation functions differ'):
                compatibility.reviewed_source(experiment)

    def test_completed_result_and_view_ids_remain_unchanged(self):
        self.prepare()
        for seed in range(1, 5000):
            target = self.pool / experiment.image_relative(seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (32, 32)).save(target)
        view = self.pool.parent / 'fid-n5000-seeds-0-4999'
        experiment.ensure_protocol(view, {'pool_protocol_id': experiment.digest(self.value), 'num_images': 5000})
        for seed in range(5000):
            experiment.link_image(self.pool / experiment.image_relative(seed), view / experiment.image_relative(seed))
        value = fixtures.ExperimentTests.result(self, self.row)
        value.update(protocol=self.value, protocol_id=experiment.digest(self.value))
        marker = self.root / 'results' / f"{self.row['condition']}-fid5000.json"
        experiment.write_json(marker, value)
        before = marker.read_bytes()
        compatibility.register(self.args, experiment)
        with mock.patch.object(experiment, 'run_logged', side_effect=AssertionError('GPU must not run')):
            experiment.evaluate(self.row, 5000, self.args, [value])
        self.assertEqual(marker.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
