"""CPU checks for narrowly approving a checkpoint selector extension."""
import json
import unittest
from unittest import mock

import extend_checkpoint_compatibility as extension
import runner_compatibility as compatibility
import run_experiment as experiment
import test_runner_compatibility as fixtures


class ExtensionTests(unittest.TestCase):
    setUp = fixtures.CompatibilityTests.setUp
    prepare = fixtures.CompatibilityTests.prepare

    def fixture(self):
        self.prepare()
        self.previous = b'CHECKPOINTS = ("20k", "40k", "60k")\ndef generate_command():\n    return 1\n'
        self.current = b'CHECKPOINTS = ("20k", "40k", "60k", "80k", "100k")\ndef generate_command():\n    return 1\n'
        self.runner = self.root / 'runner.py'
        self.runner.write_bytes(self.previous)
        self.files = dict(compatibility.files(experiment), runner=self.runner)
        patch = mock.patch.object(compatibility, 'files', return_value=self.files)
        patch.start()
        self.addCleanup(patch.stop)
        compatibility.register(self.args, experiment)
        self.record = self.root / compatibility.DIRECTORY / 'registration.json'
        self.original_record = self.record.read_bytes()
        self.runner.write_bytes(self.current)

    def test_extension_is_idempotent_and_preserves_protocol(self):
        self.fixture()
        before = (self.pool / '.protocol.json').read_bytes()
        extension.upgrade(self.args)
        self.assertEqual((self.pool / '.protocol.json').read_bytes(), before)
        self.assertEqual(experiment.protocol(self.row, self.args), self.value)
        record = json.loads(self.record.read_text())
        history = self.root / 'runner_compatibility_history' / record['previous_runner']['sha256']
        self.assertEqual((history / 'registration.json').read_bytes(), self.original_record)
        updated = self.record.read_bytes()
        extension.upgrade(self.args)
        self.assertEqual(self.record.read_bytes(), updated)

    def test_calculation_change_is_rejected_without_metadata_changes(self):
        self.fixture()
        self.runner.write_bytes(self.current.replace(b'return 1', b'return 2'))
        with self.assertRaisesRegex(ValueError, 'exceed checkpoint'):
            extension.upgrade(self.args)
        self.assertEqual(self.record.read_bytes(), self.original_record)
        self.assertEqual((self.root / compatibility.DIRECTORY / 'runner.source').read_bytes(), self.previous)

    def test_dependency_change_is_rejected(self):
        self.fixture()
        changed = self.root / 'changed-helper.py'
        changed.write_bytes(b'changed helper')
        self.files['compatibility'] = changed
        with self.assertRaisesRegex(ValueError, 'Unreviewed dependency'):
            extension.upgrade(self.args)
        self.assertEqual(self.record.read_bytes(), self.original_record)


if __name__ == '__main__':
    unittest.main()
