#!/usr/bin/env python3
"""Train experiment 0-7 using a fresh, durably reserved seed on every launch."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASE = REPO / 'experiments/0/1_cifar10_cond_edm_10k_retraining/training-runs/cifar10-cond-ddpmpp-edm-seed0-10k'
TARGETS = list(range(210000, 300001, 10000))


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_history(path):
    if not path.exists():
        return []
    # A damaged ledger must never silently free an already reserved seed.
    return [json.loads(line) for line in path.read_text().splitlines()]


def append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(timestamp=now(), **event), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def choose_seed(start, history):
    used = {0} | {row['seed'] for row in history if 'seed' in row}
    seed = start
    while seed in used:
        seed += 1
    if seed >= 2 ** 31:
        raise ValueError('Training seed range exhausted')
    return seed


def pair(directory, kimg):
    return (directory / f'network-snapshot-{kimg:06d}.pkl',
            directory / f'training-state-{kimg:06d}.pt')


def validate_pair(directory, kimg):
    import pickle
    import torch
    # Validation is CPU-only; avoid thread-pool overhead for many small tensors.
    torch.set_num_threads(1)
    snapshot_path, state_path = pair(directory, kimg)
    for path in (snapshot_path, state_path):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty checkpoint: {path}')
    with snapshot_path.open('rb') as stream:
        snapshot = pickle.load(stream)
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    state = torch.load(state_path, **kwargs)
    ema, net = snapshot['ema'], state['net']
    ema.cpu()
    net.cpu()
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not all(finite(value) for value in (ema.state_dict(), net.state_dict(), state['optimizer_state'])):
        raise ValueError(f'Nonfinite checkpoint tensors at {kimg}')
    a, b = ema.state_dict(), net.state_dict()
    if a.keys() != b.keys() or any(a[key].shape != b[key].shape for key in a):
        raise ValueError('EMA/net structure mismatch')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer.load_state_dict(state['optimizer_state'])
    if not optimizer.state:
        raise ValueError('Empty optimizer state')
    for parameter, values in optimizer.state.items():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key not in values or values[key].shape != parameter.shape:
                raise ValueError('Invalid Adam moment shape')
    return dict(kimg=kimg, snapshot=str(snapshot_path), state=str(state_path),
                snapshot_bytes=snapshot_path.stat().st_size, state_bytes=state_path.stat().st_size)


def select_resume(run_dir, validator=validate_pair, base=BASE):
    indices = set()
    for pattern in ('network-snapshot-*.pkl', 'training-state-*.pt'):
        for path in run_dir.glob(pattern):
            kimg = int(path.stem.rsplit('-', 1)[1])
            if kimg not in TARGETS:
                raise ValueError(f'Unexpected checkpoint: {path}')
            indices.add(kimg)
    rejected = []
    for kimg in sorted(indices, reverse=True):
        try:
            row = validator(run_dir, kimg)
            return kimg, Path(row['state']), rejected
        except Exception as error:
            rejected.append(dict(kimg=kimg, reason=str(error)))
    row = validator(base, 200000)
    return 200000, Path(row['state']), rejected


def validate_base():
    value = json.loads((BASE / 'training_options.json').read_text())
    expected = {'seed': 0, 'total_kimg': 200000, 'batch_size': 512,
                'batch_gpu': 128, 'ema_halflife_kimg': 500}
    sections = {
        'dataset_kwargs': {'path': 'datasets/cifar10-32x32.zip', 'max_size': 50000,
                           'use_labels': True, 'xflip': False, 'resolution': 32},
        'network_kwargs': {'class_name': 'training.networks.EDMPrecond', 'model_type': 'SongUNet',
                           'model_channels': 128, 'channel_mult': [2, 2, 2],
                           'dropout': 0.13, 'use_fp16': False},
        'optimizer_kwargs': {'class_name': 'torch.optim.Adam', 'lr': 0.001,
                             'betas': [0.9, 0.999], 'eps': 1e-8},
        'loss_kwargs': {'class_name': 'training.loss.EDMLoss'},
        'augment_kwargs': {'p': 0.12},
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('Base training settings differ')
    for section, required in sections.items():
        if any(value.get(section, {}).get(key) != item for key, item in required.items()):
            raise ValueError(f'Base training settings differ: {section}')
    config = [int(v) for v in (ROOT / 'config/checkpoint_kimgs.txt').read_text().split()]
    if config != TARGETS:
        raise ValueError('Checkpoint configuration must contain exactly 210k through 300k')
    return value


def command(run_dir, state, seed, start, dry_run=False):
    args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1']
    if dry_run:
        args = [sys.executable]
    args += ['train.py', f'--outdir={run_dir}', '--nosubdir', '--data=datasets/cifar10-32x32.zip',
             '--cond=1', '--arch=ddpmpp', '--precond=edm', '--batch=512', '--batch-gpu=128',
             '--lr=0.001', '--ema=0.5', '--dropout=0.13', '--augment=0.12', '--fp16=0',
             f'--seed={seed}', '--tick=50', '--snap=0', '--dump=0', '--duration=300',
             '--checkpoint-kimgs=' + ','.join(str(v) for v in TARGETS if v > start),
             f'--resume={state}']
    if dry_run:
        args += ['--dry-run']
    return args


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_manifest(root, create):
    files = [BASE / 'training_options.json', *pair(BASE, 200000),
             REPO / 'datasets/cifar10-32x32.zip', REPO / 'train.py',
             REPO / 'scripts/load_gpu_config.sh', REPO / 'scripts/wait_for_gpu_and_run.sh']
    for directory in ('training', 'torch_utils', 'dnnlib'):
        files.extend(sorted((REPO / directory).rglob('*.py')))
    files.extend(sorted((ROOT / 'scripts').glob('*.py')))
    files.extend(sorted((ROOT / 'scripts').glob('*.sh')))
    files.append(ROOT / 'config/checkpoint_kimgs.txt')
    value = dict(experiment='0-7', base_options=validate_base(),
                 sha256={str(p.relative_to(REPO)): identity(p) for p in files})
    path = root / 'results/experiment_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Code/data/base checkpoint changed since the first launch')
    elif create:
        if read_history(root / 'results/launch_history.jsonl'):
            raise ValueError('Existing launch history without manifest')
        write_json(path, value)


def read_progress(run_dir, offset=0):
    path = run_dir / 'stats.jsonl'
    if not path.exists():
        return None, None
    progress, loss = None, None
    with path.open() as stream:
        stream.seek(offset)
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Interrupted final writes are not valid progress evidence.
                continue
            progress = row['Progress/kimg']['mean']
            loss = row['Loss/loss']['mean']
            if not math.isfinite(progress) or not math.isfinite(loss):
                raise ValueError('Nonfinite training progress/loss')
    return progress, loss


def verify(root, run_dir):
    rows, errors = [], []
    for kimg in TARGETS:
        try:
            rows.append(validate_pair(run_dir, kimg))
        except Exception as error:
            errors.append(dict(kimg=kimg, reason=str(error)))
    progress, loss = read_progress(run_dir)
    history = read_history(root / 'results/launch_history.jsonl')
    launches = [row for row in history if row.get('event') == 'start']
    complete = (not errors and bool(launches) and progress is not None
                and 300000 <= progress < 300000.512)
    report = dict(timestamp=now(), complete=complete, actual_kimg=progress, loss=loss,
                  checkpoints=rows, rejected=errors, launch_count=len(launches))
    write_json(root / 'results/health.json', report)
    path = root / 'results/checkpoint_manifest.tsv'
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['kimg', 'snapshot', 'snapshot_bytes', 'state', 'state_bytes'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    return complete


def finish_docs(root):
    pending = '起動後はGPU待機・学習中として扱い、300k完了は検証結果で確認する。'
    completed = '300kまで到達し、全10点のsnapshot/state読込と有限lossを確認した。'
    for path in (root / 'README.md', REPO / 'experiments/0/README.md',
                 REPO / 'experiments/reports/rng_replay_audit.md'):
        content = path.read_text()
        if pending in content:
            path.write_text(content.replace(pending, completed))
    subprocess.run([sys.executable, 'scripts/build_experiment_index.py'], cwd=REPO, check=True)


def execute(root, run_dir):
    import torch
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != os.environ.get('GPU_INDEX')
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not os.environ.get('EXPECTED_GPU_NAME')):
        raise ValueError('Launch via run.sh with local GPU configuration')
    name = subprocess.check_output(['nvidia-smi', '-i', os.environ['GPU_INDEX'],
                                    '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if name != os.environ['EXPECTED_GPU_NAME'] or torch.cuda.device_count() != 1:
        raise ValueError('Configured GPU identity/count mismatch')
    if any(run_dir.glob('training-state-*.pt')) and not read_history(root / 'results/launch_history.jsonl'):
        raise ValueError('Existing checkpoints without seed history')
    ensure_manifest(root, create=True)
    start, state, rejected = select_resume(run_dir)
    write_json(root / 'results/resume_validation.json', dict(timestamp=now(), start_kimg=start, rejected=rejected))
    if start == 300000:
        if not verify(root, run_dir):
            raise ValueError('300k checkpoint exists but completion checks failed')
        finish_docs(root)
        return
    for row in rejected:
        destination = root / 'training-runs/quarantine' / uuid.uuid4().hex
        destination.mkdir(parents=True)
        for path in pair(run_dir, row['kimg']):
            if path.exists():
                path.rename(destination / path.name)
        write_json(destination / 'reason.json', row)
    ledger = root / 'results/launch_history.jsonl'
    history = read_history(ledger)
    if any(run_dir.glob('training-state-*.pt')) and not history:
        raise ValueError('Existing checkpoints without seed history')
    seed = choose_seed(start, history)
    launch_id = uuid.uuid4().hex
    args = command(run_dir, state, seed, start)
    runtime = dict(timestamp=now(), python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=subprocess.check_output(['nvidia-smi', '-i', os.environ['GPU_INDEX'],
                       '--query-gpu=index,name,uuid,driver_version', '--format=csv,noheader'], text=True).strip())
    write_json(root / 'results/runtime.json', runtime)
    stats = run_dir / 'stats.jsonl'
    offset = stats.stat().st_size if stats.exists() else 0
    append_event(ledger, dict(event='start', launch_id=launch_id, seed=seed, resume_state=str(state),
                             start_kimg=start, target_kimg=300000, command=args))
    code, status, failure = None, 'failed', None
    try:
        print('Launching: ' + shlex.join(args), flush=True)
        code = subprocess.call(args, cwd=REPO)
        if code:
            raise RuntimeError(f'Training exited with status {code}')
        progress, loss = read_progress(run_dir, offset)
        if progress is None or not 300000 <= progress < 300000.512:
            raise ValueError('Training did not reach 300k')
        if not verify(root, run_dir):
            raise ValueError('Checkpoint completion checks failed')
        status = 'complete'
    except BaseException as exc:
        failure = str(exc)
        raise
    finally:
        try:
            progress, loss = read_progress(run_dir, offset)
            error = None
        except Exception as exc:
            progress, loss, error = None, None, str(exc)
        append_event(ledger, dict(event='end', launch_id=launch_id, seed=seed, start_kimg=start,
                                 end_kimg=progress, loss=loss, exit_code=code, status=status, error=error or failure))
    finish_docs(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ('plan-only', 'dry-run', 'verify-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--exp-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.exp_root.resolve()
    run_dir = root / 'training-runs/cifar10-cond-ddpmpp-edm-post200k-no-replay'
    validate_base()
    if args.verify_only:
        ensure_manifest(root, create=False)
        if not verify(root, run_dir):
            raise SystemExit('Not complete; see results/health.json')
        return
    if args.plan_only or args.dry_run:
        start, state, rejected = select_resume(run_dir)
        seed = choose_seed(start, read_history(root / 'results/launch_history.jsonl'))
        print(json.dumps(dict(start_kimg=start, target_kimg=300000, seed=seed,
                              checkpoints=[v for v in TARGETS if v > start], rejected=rejected), indent=2))
        if start == 300000:
            return
        cmd = command(run_dir, state, seed, start, dry_run=args.dry_run)
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            subprocess.run(cmd, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        return
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    with (root / 'logs/runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(root, run_dir)


if __name__ == '__main__':
    main()
