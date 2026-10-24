#!/usr/bin/env python3
"""Continue experiment 0-1 beyond 200k and compare paired FID on one GPU."""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
BASE = REPO / 'experiments/0/1_cifar10_cond_edm_10k_retraining/training-runs/cifar10-cond-ddpmpp-edm-seed0-10k'
BASE_OPTIONS = BASE / 'training_options.json'
HELPER = REPO / 'experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run_experiment.py'
spec = importlib.util.spec_from_file_location('fid_helpers', HELPER)
fid_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid_tools)
SEEDS = ('0-49999', '50000-99999', '100000-149999')
TRANSIENT = {'run_dir', 'total_kimg', 'checkpoint_kimgs', 'snapshot_ticks', 'state_dump_ticks',
             'kimg_per_tick', 'resume_pkl', 'resume_state_dump', 'resume_kimg', 'stop_file'}


def read(path):
    return json.loads(path.read_text())


def run_dir(args):
    return args.exp_root / 'training-runs/cifar10-cond-ddpmpp-edm-seed0-post200k'


def pair(directory, kimg):
    return (directory / f'network-snapshot-{kimg:06d}.pkl',
            directory / f'training-state-{kimg:06d}.pt')


def checkpoints(start, target):
    if start < 200000 or start % 2500 or target <= 200000 or target % 10000 or start > target:
        raise ValueError('Start must be a 2.5k checkpoint >=200k; target must be a 10k multiple >200k')
    return list(range(start + 2500, target + 1, 2500))


def training_settings(value):
    result = {k: v for k, v in value.items() if k not in TRANSIENT}
    result['dataset_kwargs'] = dict(result['dataset_kwargs'])
    result['dataset_kwargs'].setdefault('random_seed', 0)
    return result


def validate_base(options):
    expected = {'total_kimg': 200000, 'seed': 0, 'batch_size': 512, 'batch_gpu': 128,
                'ema_halflife_kimg': 500, 'loss_scaling': 1.0, 'cudnn_benchmark': True}
    if any(options.get(k) != v for k, v in expected.items()):
        raise ValueError('Base training conditions differ from experiment 0-1')
    sections = {
        'dataset_kwargs': {'class_name': 'training.dataset.ImageFolderDataset',
                           'path': 'datasets/cifar10-32x32.zip', 'use_labels': True,
                           'xflip': False, 'max_size': 50000, 'resolution': 32, 'cache': True},
        'network_kwargs': {'class_name': 'training.networks.EDMPrecond', 'model_type': 'SongUNet',
                           'embedding_type': 'positional', 'encoder_type': 'standard',
                           'decoder_type': 'standard', 'model_channels': 128,
                           'channel_mult': [2, 2, 2], 'dropout': 0.13, 'use_fp16': False},
        'optimizer_kwargs': {'class_name': 'torch.optim.Adam', 'lr': 0.001,
                             'betas': [0.9, 0.999], 'eps': 1e-8},
        'loss_kwargs': {'class_name': 'training.loss.EDMLoss'},
        'augment_kwargs': {'p': 0.12},
    }
    for section, required in sections.items():
        if any(options.get(section, {}).get(k) != v for k, v in required.items()):
            raise ValueError(f'Base {section} does not match the fixed experiment')


def latest_state(args):
    directory = run_dir(args)
    paths = list(directory.glob('network-snapshot-*.pkl')) + list(directory.glob('training-state-*.pt'))
    indices = set()
    for path in paths:
        kimg = int(path.stem.rsplit('-', 1)[1])
        if kimg <= 200000 or kimg % 2500:
            raise ValueError(f'Unexpected checkpoint: {path}')
        indices.add(kimg)
        for member in pair(directory, kimg):
            if not member.is_file() or member.stat().st_size == 0:
                raise ValueError(f'Incomplete checkpoint pair: {member}')
    options = directory / 'training_options.json'
    if options.exists() and training_settings(read(options)) != training_settings(read(BASE_OPTIONS)):
        raise ValueError('Existing run has incompatible training settings')
    if paths and not options.exists():
        raise ValueError('Existing checkpoints have no training_options.json')
    current = max(indices, default=200000)
    if current > args.target_kimg:
        raise ValueError('Latest checkpoint exceeds target; specify the current or a later target')
    return current, pair(directory if indices else BASE, current)[1]


def manifest(args):
    options = read(BASE_OPTIONS)
    validate_base(options)
    files = [BASE_OPTIONS, *pair(BASE, 200000), REPO / 'datasets/cifar10-32x32.zip', args.fid_ref,
             REPO / 'train.py', REPO / 'generate.py', REPO / 'fid.py', Path(__file__), HELPER]
    for folder in ('training', 'torch_utils', 'dnnlib'):
        files.extend(sorted((REPO / folder).rglob('*.py')))
    files.extend([REPO / 'scripts/load_gpu_config.sh', REPO / 'scripts/wait_for_gpu_and_run.sh',
                  ROOT / 'scripts/gpu_worker.sh'])
    return {'experiment': '0-3', 'base_kimg': 200000, 'training_settings': training_settings(options),
            'files': {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): fid_tools.identity(p)
                      for p in files}}


def ensure_manifest(args, create=False):
    value = manifest(args)
    path = args.exp_root / 'results/experiment_manifest.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('Code/data/base model changed; use a new experiment root')
    elif create:
        if any(run_dir(args).glob('training-state-*.pt')) or any((args.exp_root / 'results').glob('*-fid50000-*.json')):
            raise ValueError('Existing outputs without an experiment manifest')
        fid_tools.write_json(path, value)
    return value


def training_command(args, current, state):
    return [args.torchrun, '--standalone', '--nproc_per_node=1', 'train.py',
            f'--outdir={run_dir(args)}', '--nosubdir', '--data=datasets/cifar10-32x32.zip',
            '--cond=1', '--arch=ddpmpp', '--precond=edm', '--batch=512', '--batch-gpu=128',
            '--lr=0.001', '--ema=0.5', '--dropout=0.13', '--augment=0.12', '--fp16=0',
            '--seed=0', '--tick=50', '--snap=0', '--dump=0', f'--duration={args.target_kimg / 1000:g}',
            '--checkpoint-kimgs=' + ','.join(map(str, checkpoints(current, args.target_kimg))),
            f'--resume={state}', f'--stop-file={args.exp_root / "STOP_AFTER_CURRENT_STAGE"}']


def runtime():
    import numpy
    import PIL
    import scipy
    import torch
    index = os.environ['GPU_INDEX']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != index or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('GPU configuration was not loaded')
    info = subprocess.check_output([
        'nvidia-smi', '-i', index, '--query-gpu=name,uuid,driver_version,pci.bus_id',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    if len(info) != 4 or info[0] != os.environ['EXPECTED_GPU_NAME'] or torch.cuda.device_count() != 1:
        raise ValueError('GPU identity/count mismatch')
    return dict(gpu_name=info[0], gpu_uuid=info[1], driver=info[2], pci_bus_id=info[3],
                physical_gpu=index, hostname=socket.gethostname(), python=sys.version,
                torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                numpy=numpy.__version__, scipy=scipy.__version__, pillow=PIL.__version__,
                numerical_environment={key: os.environ.get(key) for key in
                                       ('NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG')})


def bind_runtime(args):
    value = runtime()
    path = args.exp_root / 'results/runtime.json'
    if path.exists() and read(path) != value:
        raise ValueError('GPU/software environment changed; paired evaluation requires the same runtime')
    if not path.exists():
        fid_tools.write_json(path, value)
    return value


def row_for(args, cp, k):
    if cp < 200000 or cp % 10000 or cp > args.target_kimg or k not in (0, 1):
        raise ValueError('Evaluation requires a 10k checkpoint >=200k and K=0 or 1')
    directory = BASE if cp == 200000 else run_dir(args)
    actual, index = fid_tools.schedule_point(18) if k else (None, None)
    return dict(condition=f'{cp // 1000}k-k{k}', checkpoint=f'{cp // 1000}k', main_steps=18,
                restart_steps=3 if k else 0, restart_k=k, t_min=0.14 if k else 0.0,
                t_max=0.30 if k else 0.0, nfe=35 + 4 * k, actual_t_min=actual,
                restart_index=index, network=str(pair(directory, cp)[0]))


def protocol(args, row, seeds, environment):
    value = fid_tools.protocol(row, args, seeds)
    value.update(experiment='0-3', extension_runner=fid_tools.identity(Path(__file__)), runtime=environment)
    return value


def result_path(args, row, seeds):
    return args.exp_root / 'results' / f'{row["condition"]}-fid50000-seeds-{seeds}.json'


def verify_result(args, path, environment, images=True):
    result = read(path)
    row = row_for(args, result['kimg'], result['restart_k'])
    seeds = result['seeds']
    if seeds not in SEEDS or path != result_path(args, row, seeds):
        raise ValueError(f'Invalid result seed/filename: {path}')
    p = protocol(args, row, seeds, environment)
    if (result.get('protocol') != p or result.get('protocol_id') != fid_tools.digest(p)
            or result.get('num_images') != 50000 or not math.isfinite(result['fid']) or result['fid'] < 0):
        raise ValueError(f'Invalid/changed result protocol: {path}')
    if images:
        folder = args.exp_root / 'images' / row['condition'] / f'fid-n50000-seeds-{seeds}'
        fid_tools.ensure_protocol(folder, p, create=False)
        fid_tools.inventory(folder, fid_tools.parse_seeds(seeds), complete=True)
    return result


def evaluate(args, environment):
    row = row_for(args, args.checkpoint_kimg, args.restart_k)
    seeds = args.seeds
    p = protocol(args, row, seeds, environment)
    folder = args.exp_root / 'images' / row['condition'] / f'fid-n50000-seeds-{seeds}'
    marker = result_path(args, row, seeds)
    if marker.exists():
        verify_result(args, marker, environment)
        return
    fid_tools.ensure_protocol(folder, p)
    wanted = fid_tools.parse_seeds(seeds)
    existing = fid_tools.inventory(folder, wanted)
    log = args.exp_root / 'logs' / f'{row["condition"]}-{seeds}.log'
    for missing in fid_tools.seed_ranges(sorted(set(wanted) - set(existing))):
        fid_tools.run_logged(fid_tools.generate_command(row, folder, missing, args), log)
    fid_tools.inventory(folder, wanted, complete=True)
    command = [args.torchrun, '--standalone', '--nproc_per_node=1', 'fid.py', 'calc',
               f'--images={folder}', f'--ref={args.fid_ref}', '--num=50000', '--batch=64']
    _, fid = fid_tools.run_logged(command, log)
    if fid is None or not math.isfinite(fid) or fid < 0:
        raise ValueError('Missing or invalid FID')
    fid_tools.write_json(marker, dict(kimg=args.checkpoint_kimg, restart_k=args.restart_k,
                         seeds=seeds, num_images=50000, fid=fid, protocol=p,
                         protocol_id=fid_tools.digest(p), images=str(folder), log=str(log)))


def health(args, current):
    state = pair(run_dir(args), current)
    if current == 200000:
        return
    entries = [json.loads(line) for line in (run_dir(args) / 'stats.jsonl').read_text().splitlines() if line.strip()]
    last = entries[-1]
    loss = last['Loss/loss']['mean']
    progress = last['Progress/kimg']['mean']
    if not math.isfinite(loss) or not current <= progress < current + 0.512:
        raise ValueError('Nonfinite loss or checkpoint/progress mismatch')
    fid_tools.write_json(args.exp_root / 'results/health.json',
                         dict(kimg=current, actual_kimg=progress, loss=loss,
                              snapshot=fid_tools.identity(state[0]), state=fid_tools.identity(state[1])))


def screening_passes(baseline, candidate):
    # Decimal prevents 0.01 becoming 0.009999999999999787 at the boundary.
    return Decimal(str(baseline)) - Decimal(str(candidate)) >= Decimal('0.01')


def aggregate(args, results):
    lookup = {(r['kimg'], r['restart_k'], r['seeds']): r['fid'] for r in results}
    summaries, paired = [], []
    for cp in sorted({r['kimg'] for r in results} - {200000}):
        for k in (0, 1):
            diffs = []
            base, candidate = [], []
            for seed in SEEDS:
                a, b = lookup.get((200000, k, seed)), lookup.get((cp, k, seed))
                if a is not None and b is not None:
                    diffs.append(a - b)
                    base.append(a)
                    candidate.append(b)
                    paired.append(dict(kimg=cp, restart_k=k, seeds=seed, fid_200k=a,
                                       fid_candidate=b, improvement=a-b))
            if not diffs:
                continue
            complete = len(diffs) == 3
            improved = complete and statistics.mean(diffs) > 0 and sum(d > 0 for d in diffs) >= 2
            summaries.append(dict(kimg=cp, restart_k=k, n=len(diffs), complete=complete,
                                  baseline_mean=statistics.mean(base), candidate_mean=statistics.mean(candidate),
                                  baseline_std=statistics.stdev(base) if len(base) > 1 else None,
                                  candidate_std=statistics.stdev(candidate) if len(candidate) > 1 else None,
                                  improvement_mean=statistics.mean(diffs),
                                  improvement_std=statistics.stdev(diffs) if len(diffs) > 1 else None,
                                  positive_ranges=sum(d > 0 for d in diffs),
                                  assessment='improvement_tendency' if improved else
                                  'no_improvement_tendency' if complete else 'screen_only'))
    out = args.exp_root / 'results'
    fid_tools.write_json(out / 'comparison.json', summaries)
    fid_tools.write_json(out / 'paired_improvements.json', paired)
    for name, table in [('comparison', summaries), ('paired_improvements', paired)]:
        if table:
            fid_tools.write_tsv(out / f'{name}.tsv', table, list(table[0]))


def results(args, images=False):
    paths = sorted((args.exp_root / 'results').glob('*-fid50000-seeds-*.json'))
    if not paths:
        return []
    environment = read(args.exp_root / 'results/runtime.json')
    return [verify_result(args, path, environment, images=images) for path in paths]


def stopped(args):
    if (args.exp_root / 'STOP_AFTER_CURRENT_STAGE').exists():
        print('STOP_AFTER_CURRENT_STAGE exists; exiting safely. Remove it and rerun to resume.', flush=True)
        return True
    return False


def launch(args, job, cp=200000, k=0, seeds=SEEDS[0]):
    env = dict(os.environ, PYTHON=sys.executable, RUN_SCRIPT=str(ROOT / 'scripts/gpu_worker.sh'),
               EXP_ROOT=str(args.exp_root), SCHEDULE_LABEL=f'0-3 {job} {cp} K={k} {seeds}', PLAN_ONLY='0')
    command = ['bash', 'scripts/wait_for_gpu_and_run.sh', '--worker', job,
               '--exp-root', str(args.exp_root), '--target-kimg', str(args.target_kimg),
               '--fid-ref', str(args.fid_ref), '--torchrun', args.torchrun,
               '--checkpoint-kimg', str(cp), '--restart-k', str(k), '--seeds', seeds]
    print(shlex.join(command), flush=True)
    log = args.exp_root / 'logs/master.log'
    with log.open('a') as handle:
        subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-root', type=Path, default=Path(os.environ.get('EXP_ROOT', ROOT)))
    parser.add_argument('--target-kimg', type=int, default=210000)
    modes = parser.add_mutually_exclusive_group()
    for name in ('plan-only', 'verify-only', 'train-only', 'eval-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--fid-stage', choices=('screen', 'auto', 'all'), default='auto')
    parser.add_argument('--fid-ref', type=Path, default=REPO / 'fid-refs/cifar10-32x32.npz')
    parser.add_argument('--torchrun', default=os.environ.get('TORCHRUN', str(Path(sys.executable).parent / 'torchrun')))
    parser.add_argument('--worker', choices=('train', 'eval'), help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint-kimg', type=int, default=200000, help=argparse.SUPPRESS)
    parser.add_argument('--restart-k', type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    parser.add_argument('--seeds', choices=SEEDS, default=SEEDS[0], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.exp_root = args.exp_root.resolve()
    args.fid_ref = args.fid_ref.resolve()
    args.gen_batch = args.fid_batch = 64
    checkpoints(200000, args.target_kimg)
    return args


def main():
    args = arguments()
    current, state = latest_state(args)
    ensure_manifest(args)
    if args.plan_only or os.environ.get('PLAN_ONLY') == '1':
        print(f'Train {current} -> {args.target_kimg} kimg; LR=0.001; save {checkpoints(current, args.target_kimg)}')
        if current < args.target_kimg:
            print(shlex.join(training_command(args, current, state)))
        print('FID: same GPU, 200k vs target, K=0/1, FID50k seeds 0-49999.')
        print(f'Confirmation: {args.fid_stage}; auto adds two ranges if K=0 improves by >=0.01.')
        return
    if args.verify_only:
        if not (args.exp_root / 'results/experiment_manifest.json').exists():
            raise ValueError('No experiment manifest; experiment has not been initialized')
        completed = results(args, images=True)
        health_path = args.exp_root / 'results/health.json'
        if current > 200000:
            saved = read(health_path)
            snapshot, dump = pair(run_dir(args), current)
            if saved['kimg'] != current or saved['snapshot'] != fid_tools.identity(snapshot) or saved['state'] != fid_tools.identity(dump):
                raise ValueError('Health/checkpoint identity mismatch')
        print(f'Verified latest checkpoint {current}, {len(completed)} FID results; '
              f'training target reached={current == args.target_kimg}')
        return
    if args.worker:
        if stopped(args):
            return
        environment = bind_runtime(args)
        if args.worker == 'eval':
            evaluate(args, environment)
        elif current < args.target_kimg:
            fid_tools.run_logged(training_command(args, current, state), args.exp_root / 'logs/train.log')
            current, _ = latest_state(args)
            health(args, current)
            if current < args.target_kimg and not stopped(args):
                raise ValueError('Training exited before the target without a STOP request')
        return
    args.exp_root.mkdir(parents=True, exist_ok=True)
    (args.exp_root / 'logs').mkdir(exist_ok=True)
    with (args.exp_root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_manifest(args, create=True)
        if stopped(args):
            return
        if not args.eval_only and current < args.target_kimg:
            launch(args, 'train')
        current, _ = latest_state(args)
        if stopped(args):
            return
        if current != args.target_kimg:
            raise ValueError('Target checkpoint is missing; train first')
        health(args, current)
        if args.train_only:
            return
        # Always rerun the baseline on this GPU; never import cross-GPU FID values.
        for seed in SEEDS:
            if seed != SEEDS[0]:
                lookup = {(r['kimg'], r['restart_k'], r['seeds']): r['fid'] for r in results(args)}
                passed = screening_passes(lookup[(200000, 0, SEEDS[0])], lookup[(args.target_kimg, 0, SEEDS[0])])
                fid_tools.write_json(args.exp_root / f'results/screening-{args.target_kimg}.json',
                                     dict(target_kimg=args.target_kimg, threshold=0.01, passed=passed,
                                          fid_stage=args.fid_stage, confirmation_requested=args.fid_stage == 'all' or (args.fid_stage == 'auto' and passed)))
                if args.fid_stage == 'screen' or (args.fid_stage == 'auto' and not passed):
                    print('Screening complete; no further seed ranges requested.', flush=True)
                    break
            for cp in (200000, args.target_kimg):
                for k in (0, 1):
                    if stopped(args):
                        return
                    marker = result_path(args, row_for(args, cp, k), seed)
                    if marker.exists():
                        verify_result(args, marker, read(args.exp_root / 'results/runtime.json'))
                    else:
                        launch(args, 'eval', cp, k, seed)
                        if stopped(args) and not marker.exists():
                            return
                    aggregate(args, results(args))
        print(f'Finished {args.target_kimg} kimg; see {args.exp_root / "results/comparison.tsv"}', flush=True)


if __name__ == '__main__':
    main()
