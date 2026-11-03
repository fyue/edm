"""CPU-only approval of the exact 20k..60k -> 20k..100k selector extension."""
import argparse
import ast
import copy
import datetime
import fcntl
import json
from pathlib import Path
import tempfile

import run_experiment as experiment
import runner_compatibility as compatibility


OLD_CHECKPOINTS = ('20k', '40k', '60k')
NEW_CHECKPOINTS = (*OLD_CHECKPOINTS, '80k', '100k')
MESSAGES = {
    'Require 20k,40k,60k,80k,100k x K=0..8': 'Require 20k,40k,60k x K=0..8',
    'Incomplete 45-condition configuration': 'Incomplete 27-condition configuration',
    'Checkpoint outside 20k,40k,60k,80k,100k: ': 'Checkpoint outside 20k,40k,60k: ',
}


def check_extension(previous, current):
    old_tree, new_tree = ast.parse(previous), ast.parse(current)
    old_assign = next(node for node in old_tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CHECKPOINTS' for t in node.targets))
    new_assign = next(node for node in new_tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CHECKPOINTS' for t in node.targets))
    if ast.literal_eval(old_assign.value) != OLD_CHECKPOINTS or ast.literal_eval(new_assign.value) != NEW_CHECKPOINTS:
        raise ValueError('Not the supported checkpoint-only extension')
    new_assign.value = copy.deepcopy(old_assign.value)

    class Messages(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in MESSAGES:
                node.value = MESSAGES[node.value]
            return node

    new_tree = Messages().visit(new_tree)
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError('Runner changes exceed checkpoint selection/error messages; refusing approval')


def upgrade(args):
    root = args.exp_root.resolve()
    directory = root / compatibility.DIRECTORY
    with (root / 'logs/.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((directory / 'registration.json').read_text())
        if record['root'] != str(root) or record['experiment'] != experiment.EXPERIMENT_ID:
            raise ValueError('Registration belongs to another output root or mode')
        if record['legacy_runner'] != compatibility.LEGACY:
            raise ValueError('Unknown legacy compatibility basis')
        actual = compatibility.files(experiment)
        current = actual['runner'].read_bytes()
        if compatibility.content_identity(current) == record['approved_files']['runner']:
            compatibility.read_registration(root, experiment)
            print('Checkpoint extension already approved; no changes')
            return
        history = root / 'runner_compatibility_history' / record['approved_files']['runner']['sha256']
        sources = {}
        for name, expected in record['approved_files'].items():
            if name != 'runner' and compatibility.content_identity(actual[name].read_bytes()) != expected:
                raise ValueError(f'Unreviewed dependency change: {name}')
            candidates = [directory / f'{name}.source', history / f'{name}.source']
            source = next((p for p in candidates if p.exists() and compatibility.content_identity(p.read_bytes()) == expected), None)
            if source is None:
                raise ValueError(f'Missing original source snapshot: {name}')
            sources[name] = source.read_bytes()
        legacy = (directory / 'legacy_runner.source').read_bytes()
        if compatibility.content_identity(legacy) != compatibility.LEGACY:
            raise ValueError('Legacy snapshot changed')
        check_extension(sources['runner'], current)
        updated = copy.deepcopy(record)
        updated['approved_files']['runner'] = compatibility.content_identity(current)
        updated['registered_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        updated['previous_runner'] = record['approved_files']['runner']
        updated['upgrade_reason'] = 'Checkpoint selection extended to 80k/100k; no calculation changes'
        # Validate through a temporary registration without rewriting real protocols.
        with tempfile.TemporaryDirectory(prefix='edm814-extension-') as temporary:
            temporary = Path(temporary).resolve()
            for name in ('images', 'results', 'progress'):
                (temporary / name).symlink_to(root / name, target_is_directory=True)
            provisional = temporary / compatibility.DIRECTORY
            provisional.mkdir()
            for name, data in dict(sources, runner=current, legacy_runner=legacy).items():
                (provisional / f'{name}.source').write_bytes(data)
            candidate = dict(updated, root=str(temporary))
            experiment.write_json(provisional / 'registration.json', candidate)
            check_args = argparse.Namespace(exp_root=temporary, config=args.config,
                                            fid_ref=args.fid_ref.resolve(), gen_batch=64, fid_batch=64)
            baseline, networks = compatibility.validate_existing(check_args, experiment)
            if baseline != record['baseline_common'] or any(networks.get(k) != v for k, v in record['networks'].items()):
                raise ValueError('Existing computation identities changed')
        history.mkdir(parents=True, exist_ok=True)
        for name, data in dict(sources, legacy_runner=legacy).items():
            path = history / f'{name}.source'
            if path.exists() and path.read_bytes() != data:
                raise ValueError('History snapshot differs')
            if not path.exists():
                path.write_bytes(data)
        path = history / 'registration.json'
        if not path.exists():
            experiment.write_json(path, record)
        temporary_source = directory / 'runner.source.tmp'
        temporary_source.write_bytes(current)
        temporary_source.replace(directory / 'runner.source')
        experiment.write_json(directory / 'registration.json', updated)
        compatibility.read_registration(root, experiment)
        print(f'Approved checkpoint-only extension; previous registration saved in {history}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-root', type=Path, default=experiment.DEFAULT_ROOT / 'runs' / experiment.MODE)
    parser.add_argument('--config', type=Path, default=experiment.DEFAULT_ROOT / 'config' / experiment.CONFIG_NAME)
    parser.add_argument('--fid-ref', type=Path, default=experiment.REPO_ROOT / 'fid-refs/cifar10-32x32.npz')
    upgrade(parser.parse_args())


if __name__ == '__main__':
    main()
