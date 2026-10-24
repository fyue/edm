#!/usr/bin/env python3
"""Experiment 0-4: paired K=0/1 FID-50k at 200k/210k over three seed ranges."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE = REPO / 'experiments/0/3_cifar10_cond_edm_post200k_extension'
spec = importlib.util.spec_from_file_location('experiment03', SOURCE / 'scripts/run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
util = base.fid_tools
SEEDS = base.SEEDS
CHECKPOINTS = ('200k', '210k')


def read(path):
    return json.loads(path.read_text())


def definitions():
    # Avoid the training coordinator: 220k states may already exist in 0-3.
    args = SimpleNamespace(exp_root=SOURCE, target_kimg=220000)
    return [base.row_for(args, cp, k) for cp in (200000, 210000) for k in (0, 1)]


def select(values, text):
    if text == 'all':
        return list(values)
    requested = text.replace(',', ' ').split()
    if not requested or set(requested) - set(values) or len(set(requested)) != len(requested):
        raise ValueError(f'Invalid/duplicate selection {text}; choices: {list(values)}')
    return [v for v in values if v in requested]


def marker(args, row, seeds):
    return args.exp_root / 'results' / f'{row["condition"]}-fid50000-seeds-{seeds}.json'


def folder_for(args, row, seeds, smoke=False):
    return args.exp_root / ('samples' if smoke else 'images') / row['condition'] / (
        'seeds-0-1' if smoke else f'fid-n50000-seeds-{seeds}')


def sources(args):
    environment = read(SOURCE / 'results/runtime.json')
    original_args = SimpleNamespace(exp_root=SOURCE, target_kimg=220000,
                                    fid_ref=args.fid_ref, gen_batch=64, fid_batch=64)
    found = {}
    for row in definitions():
        path = SOURCE / 'results' / marker(args, row, SEEDS[0]).name
        result = base.verify_result(original_args, path, environment, images=False)
        found[row['condition']] = (path, result)
    return environment, found


def protocol(args, row, seeds, environment):
    p = base.protocol(args, row, seeds, environment)
    p.update(experiment='0-4', evaluation_runner=util.identity(Path(__file__)),
             gpu_worker=util.identity(ROOT / 'scripts/gpu_worker.sh'),
             gpu_waiter=util.identity(REPO / 'scripts/wait_for_gpu_and_run.sh'),
             gpu_loader=util.identity(REPO / 'scripts/load_gpu_config.sh'))
    return p


def source_protocol(p):
    result = {k: v for k, v in p.items() if k not in
              ('evaluation_runner', 'gpu_worker', 'gpu_waiter', 'gpu_loader')}
    result['experiment'] = '0-3'
    return result


def imported(args, row, path, source, environment):
    p = protocol(args, row, SEEDS[0], environment)
    if source.get('protocol') != source_protocol(p) or source.get('protocol_id') != util.digest(source['protocol']):
        raise ValueError('Source protocol/model/code/reference/runtime mismatch')
    return dict(row, seeds=SEEDS[0], num_images=50000, fid=source['fid'],
                protocol=p, protocol_id=util.digest(p), origin='0-3', source_path=str(path),
                source_result=source, images='', generate_sec=0.0, fid_sec=0.0)


def validate(args, results, environment, images=False):
    definitions_by_name = {r['condition']: r for r in definitions()}
    seen = set()
    for result in results:
        row = definitions_by_name.get(result.get('condition'))
        seed = result.get('seeds')
        key = result.get('condition'), seed
        if key in seen:
            raise ValueError(f'Duplicate completed result: {key}')
        seen.add(key)
        if row is None or seed not in SEEDS or any(result.get(k) != v for k, v in row.items()):
            raise ValueError('Unknown or modified result condition/seed')
        p = protocol(args, row, seed, environment)
        if (result.get('protocol') != p or result.get('protocol_id') != util.digest(p)
                or result.get('num_images') != 50000 or not isinstance(result.get('fid'), (int, float))
                or not math.isfinite(result['fid']) or result['fid'] < 0):
            raise ValueError('Completed result protocol/FID mismatch')
        if seed == SEEDS[0]:
            source = result.get('source_result', {})
            expected_path = SOURCE / 'results' / marker(args, row, seed).name
            if (result.get('origin') != '0-3' or result.get('source_path') != str(expected_path)
                    or source != read(expected_path) or source.get('protocol') != source_protocol(p)
                    or source.get('protocol_id') != util.digest(source['protocol'])
                    or source.get('fid') != result['fid'] or result.get('images') != ''):
                raise ValueError('Imported source/protocol/FID mismatch')
        else:
            folder = folder_for(args, row, seed)
            if (result.get('origin') != 'generated' or result.get('images') != str(folder)
                    or result.get('source_path') != '' or result.get('source_result') is not None):
                raise ValueError('Invalid generated-result provenance')
            if images:
                util.ensure_protocol(folder, p, create=False)
                util.inventory(folder, util.parse_seeds(seed), complete=True)


def load_results(args, environment, images=False):
    results = []
    for path in sorted((args.exp_root / 'results').glob('*-fid50000-seeds-*.json')):
        result = read(path)
        if path != marker(args, result, result['seeds']):
            raise ValueError('Result filename mismatch')
        results.append(result)
    validate(args, results, environment, images=images)
    return results


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None, complete=len(values) == 3)


def aggregate(args, results, environment):
    validate(args, results, environment)
    out = args.exp_root / 'results'
    lookup = {(r['condition'], r['seeds']): r['fid'] for r in results}
    raw = [{k: r[k] for k in ('condition', 'checkpoint', 'restart_k', 'nfe', 'seeds', 'fid', 'origin')}
           for r in sorted(results, key=lambda r: (r['checkpoint'], r['restart_k'], SEEDS.index(r['seeds'])))]
    util.write_json(out / 'restart_fid_summary.json', results)
    util.write_tsv(out / 'restart_fid_summary.tsv', raw,
                   ['condition', 'checkpoint', 'restart_k', 'nfe', 'seeds', 'fid', 'origin'])
    conditions, paired, improvements, training, training_stats = [], [], [], [], []
    for row in definitions():
        values = [lookup[(row['condition'], seed)] for seed in SEEDS if (row['condition'], seed) in lookup]
        conditions.append(dict(condition=row['condition'], checkpoint=row['checkpoint'],
                               restart_k=row['restart_k'], nfe=row['nfe'], **stats(values)))
    for cp in CHECKPOINTS:
        differences = []
        for seed in SEEDS:
            a, b = lookup.get((f'{cp}-k0', seed)), lookup.get((f'{cp}-k1', seed))
            if a is not None and b is not None:
                differences.append(a - b)
                paired.append(dict(checkpoint=cp, seeds=seed, fid_k0=a, fid_k1=b, improvement=a-b))
        improvements.append(dict(checkpoint=cp, positive_ranges=sum(v > 0 for v in differences), **stats(differences)))
    for k in (0, 1):
        differences = []
        for seed in SEEDS:
            a, b = lookup.get((f'200k-k{k}', seed)), lookup.get((f'210k-k{k}', seed))
            if a is not None and b is not None:
                differences.append(b - a)
                training.append(dict(restart_k=k, seeds=seed, fid_200k=a, fid_210k=b, delta_210k_minus_200k=b-a))
        training_stats.append(dict(restart_k=k, worsened_ranges=sum(v > 0 for v in differences), **stats(differences)))
    for name, table, fields in [
            ('condition_stats', conditions, list(conditions[0])),
            ('paired_improvements', paired, ['checkpoint', 'seeds', 'fid_k0', 'fid_k1', 'improvement']),
            ('improvement_stats', improvements, list(improvements[0])),
            ('training_deltas', training, ['restart_k', 'seeds', 'fid_200k', 'fid_210k', 'delta_210k_minus_200k']),
            ('training_delta_stats', training_stats, list(training_stats[0]))]:
        util.write_json(out / f'{name}.json', table)
        util.write_tsv(out / f'{name}.tsv', table, fields)
    util.write_json(out / 'status.json', dict(completed=len(results), expected=12, complete=len(results) == 12,
                                             note='Fixed main=18; K=0:35 NFE, K=1:39 NFE; generation seed ranges, not retraining seeds'))
    plot(out, lookup)


def plot(out, lookup):
    os.environ.setdefault('MPLCONFIGDIR', str(out.parent / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, cp in enumerate(CHECKPOINTS):
        for k, offset, color in ((0, -.12, 'tab:gray'), (1, .12, 'tab:blue')):
            values = [lookup[(f'{cp}-k{k}', s)] for s in SEEDS if (f'{cp}-k{k}', s) in lookup]
            axes[0].scatter([i+offset]*len(values), values, color=color,
                            label=f'K={k} ({35+4*k} NFE)' if i == 0 else None)
            if len(values) == 3:
                axes[0].errorbar(i+offset, statistics.mean(values), yerr=statistics.stdev(values),
                                 fmt='s', color=color, capsize=4)
        gains = [lookup[(f'{cp}-k0', s)] - lookup[(f'{cp}-k1', s)] for s in SEEDS
                 if (f'{cp}-k0', s) in lookup and (f'{cp}-k1', s) in lookup]
        axes[1].scatter([i]*len(gains), gains, color='tab:blue')
        if len(gains) == 3:
            axes[1].errorbar(i, statistics.mean(gains), yerr=statistics.stdev(gains), fmt='s', color='black', capsize=4)
    for k in (0, 1):
        deltas = [lookup[(f'210k-k{k}', s)] - lookup[(f'200k-k{k}', s)] for s in SEEDS
                  if (f'210k-k{k}', s) in lookup and (f'200k-k{k}', s) in lookup]
        axes[2].scatter([k]*len(deltas), deltas, color='tab:orange')
        if len(deltas) == 3:
            axes[2].errorbar(k, statistics.mean(deltas), yerr=statistics.stdev(deltas), fmt='s', color='black', capsize=4)
    axes[0].set(ylabel='FID-50k', title='Fixed main=18: K=0 versus K=1')
    axes[0].legend()
    axes[1].set(ylabel='FID(K=0) - FID(K=1)', title='Restart gain (positive = improvement)')
    axes[2].set(ylabel='FID(210k) - FID(200k)', title='Training extension (positive = worse)')
    for axis, labels in zip(axes, (CHECKPOINTS, CHECKPOINTS, ('K=0', 'K=1'))):
        axis.set_xticks([0, 1])
        axis.set_xticklabels(labels)
    for axis in axes[1:]:
        axis.axhline(0, color='gray', linewidth=1)
    fig.suptitle('Points: 50k generation seed ranges; bars: mean +/- sample SD only at n=3')
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'fid50k_3seeds.{suffix}', dpi=180)
    plt.close(fig)


def stopped(args):
    path = args.exp_root / 'STOP_AFTER_CURRENT_CONDITION'
    if path.exists():
        print(f'STOP: {path}; remove it and rerun to resume.', flush=True)
        return True
    return False


def evaluate(args, row, seeds, environment, smoke=False):
    seeds = '0-1' if smoke else seeds
    folder = folder_for(args, row, seeds, smoke)
    p = protocol(args, row, seeds, environment)
    util.ensure_protocol(folder, p)
    allowed = util.parse_seeds(seeds)
    present = util.inventory(folder, allowed)
    generate_sec = 0.0
    log = args.exp_root / 'logs' / f'{row["condition"]}-seeds-{seeds}.log'
    for missing in util.seed_ranges(sorted(set(allowed) - set(present))):
        elapsed, _ = util.run_logged(util.generate_command(row, folder, missing, args), log)
        generate_sec += elapsed
    util.inventory(folder, allowed, complete=True)
    if smoke:
        print(f'SMOKE OK {row["condition"]}: two valid RGB 32x32 PNGs', flush=True)
        return
    command = [args.torchrun, '--standalone', '--nproc_per_node=1', 'fid.py', 'calc',
               f'--images={folder}', f'--ref={args.fid_ref}', '--num=50000', '--batch=64']
    fid_sec, fid = util.run_logged(command, log)
    if fid is None or not math.isfinite(fid) or fid < 0:
        raise ValueError('Missing/nonfinite/negative FID')
    result = dict(row, seeds=seeds, num_images=50000, fid=fid, protocol=p, protocol_id=util.digest(p),
                  origin='generated', source_path='', source_result=None, images=str(folder),
                  generate_sec=generate_sec, fid_sec=fid_sec)
    validate(args, [result], environment)
    util.write_json(marker(args, row, seeds), result)


def launch(args, row, seeds):
    env = dict(os.environ, PYTHON=sys.executable, RUN_SCRIPT=str(ROOT / 'scripts/gpu_worker.sh'),
               EXP_ROOT=str(args.exp_root), SCHEDULE_LABEL=f'0-4 {row["condition"]} {seeds}', PLAN_ONLY='0')
    command = ['bash', 'scripts/wait_for_gpu_and_run.sh', '--gpu-worker', row['condition'],
               '--exp-root', str(args.exp_root), '--fid-ref', str(args.fid_ref),
               '--torchrun', args.torchrun, '--seed-filter', seeds]
    if args.samples_only:
        command.append('--samples-only')
    # Preserve progress in tmux and master.log, including shared-waiter errors.
    with subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        with (args.exp_root / 'logs/master.log').open('a') as handle:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
        if process.wait():
            raise RuntimeError('GPU stage failed; see logs/master.log')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ('plan-only', 'verify-only', 'aggregate-only', 'samples-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--exp-root', type=Path, default=Path(os.environ.get('EXP_ROOT', ROOT)))
    parser.add_argument('--checkpoint-filter', default='all')
    parser.add_argument('--seed-filter', default='all')
    parser.add_argument('--fid-ref', type=Path, default=REPO / 'fid-refs/cifar10-32x32.npz')
    parser.add_argument('--torchrun', default=os.environ.get('TORCHRUN', str(Path(sys.executable).parent / 'torchrun')))
    parser.add_argument('--gpu-worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.exp_root = args.exp_root.resolve()
    args.fid_ref = args.fid_ref.resolve()
    args.gen_batch = args.fid_batch = 64
    return args


def main():
    args = arguments()
    selected_cps = select(CHECKPOINTS, args.checkpoint_filter)
    selected_seeds = select(SEEDS, args.seed_filter)
    rows = [r for r in definitions() if r['checkpoint'] in selected_cps]
    environment, originals = sources(args)
    results = load_results(args, environment)
    if args.plan_only or os.environ.get('PLAN_ONLY') == '1':
        jobs = [(r, s) for r in rows for s in selected_seeds]
        print(f'{len(jobs)} evaluations selected; initial full plan: 4 reused + 8 new = 12; 400000 new images')
        for row, seed in jobs:
            print(row['condition'], f'NFE={row["nfe"]}', seed,
                  'import 0-3' if seed == SEEDS[0] else 'generate', flush=True)
        return
    if args.verify_only:
        load_results(args, environment, images=True)
        print(f'Verified {len(results)}/12 evaluations')
        return
    if args.gpu_worker:
        candidates = [r for r in definitions() if r['condition'] == args.gpu_worker]
        if len(candidates) != 1 or len(selected_seeds) != 1 or (selected_seeds[0] == SEEDS[0] and not args.samples_only):
            raise ValueError('Invalid worker condition/seed')
        if stopped(args):
            return
        if base.runtime() != environment:
            raise ValueError('GPU/software runtime differs from original 0-3 evaluations')
        evaluate(args, candidates[0], selected_seeds[0], environment, smoke=args.samples_only)
        return
    args.exp_root.mkdir(parents=True, exist_ok=True)
    with (args.exp_root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.exp_root / 'logs').mkdir(exist_ok=True)
        if args.aggregate_only:
            aggregate(args, results, environment)
            print(f'Aggregated {len(results)}/12 evaluations')
            return
        if stopped(args):
            return
        if not args.samples_only:
            known = {(r['condition'], r['seeds']) for r in results}
            for row in rows:
                if SEEDS[0] in selected_seeds and (row['condition'], SEEDS[0]) not in known:
                    path, original = originals[row['condition']]
                    result = imported(args, row, path, original, environment)
                    validate(args, [result], environment)
                    util.write_json(marker(args, row, SEEDS[0]), result)
                    results.append(result)
            aggregate(args, results, environment)
        for row in rows:
            for seed in (SEEDS[:1] if args.samples_only else selected_seeds):
                if stopped(args):
                    return
                if not args.samples_only and seed == SEEDS[0]:
                    continue
                path = marker(args, row, seed)
                if not args.samples_only and path.exists():
                    validate(args, [read(path)], environment, images=True)
                    continue
                launch(args, row, seed)
                if stopped(args) and not path.exists():
                    return
                if not args.samples_only:
                    results = load_results(args, environment)
                    aggregate(args, results, environment)
        print('Selected evaluations finished.', flush=True)


if __name__ == '__main__':
    main()
