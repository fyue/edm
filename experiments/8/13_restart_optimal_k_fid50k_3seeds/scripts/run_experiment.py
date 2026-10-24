#!/usr/bin/env python3
"""Experiment 8-13: fixed optimal K versus K=0, FID-50k over three seed ranges."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE = REPO / 'experiments/8/11_restart_fixed_main18_tmin014_k_sweep'
sys.path.insert(0, str(REPO / 'scripts'))
from condition_pause import ConditionPause, StopRequested

spec = importlib.util.spec_from_file_location('experiment811', SOURCE / 'scripts/run_experiment.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
OPTIMAL = {'20k': 7, '60k': 3, '100k': 3, '140k': 1, '180k': 1, '200k': 1}
SEEDS = ('0-49999', '50000-99999', '100000-149999')
SOURCE_ROOTS = (SOURCE, SOURCE / 'runs/20k-140k')


def conditions():
    original = {r['condition']: r for r in old.load_conditions(SOURCE / 'config/restart_fixed_main18_tmin014.tsv')}
    configured = old.read_tsv(ROOT / 'config/conditions.tsv')
    expected = {f'{cp}-k{k}' for cp, best in OPTIMAL.items() for k in (0, best)}
    if len(configured) != 12 or {r['condition'] for r in configured} != expected:
        raise ValueError('Require the twelve fixed checkpoint/K conditions')
    rows = []
    for row in configured:
        reference = original[row['condition']]
        for key, value in row.items():
            actual = str((REPO / value).resolve()) if key == 'network' else value
            if key in ('main_steps', 'restart_steps', 'restart_k', 'nfe'):
                actual = int(actual)
            elif key in ('t_min', 't_max'):
                actual = float(actual)
            if actual != reference[key]:
                raise ValueError(f'Condition differs from 8-11: {row["condition"]} {key}')
        rows.append(reference)
    return rows


def select(values, text):
    if not text or text == 'all':
        return list(values)
    requested = text.replace(',', ' ').split()
    if not requested or set(requested) - set(values):
        raise ValueError(f'Invalid selection: {text}; choices: {list(values)}')
    return [v for v in values if v in requested]


def key(row):
    return row['condition'], row['seeds']


def marker(root, row):
    return root / 'results' / f'{row["condition"]}-fid50000-seeds-{row["seeds"]}.json'


def protocol(row, seeds, args):
    value = old.protocol(row, args, seeds)
    value['experiment'] = '8-13'
    value['runner'] = old.identity(Path(__file__))
    value['base_runner'] = old.identity(SOURCE / 'scripts/run_experiment.py')
    return value


def sampling_identity(value):
    return {k: v for k, v in value.items() if k not in ('experiment', 'runner', 'base_runner')}


def legacy_results(rows):
    wanted = {r['condition'] for r in rows}
    results = []
    for condition in wanted:
        paths = [r / 'results' / f'{condition}-fid50000.json' for r in SOURCE_ROOTS]
        found = [p for p in paths if p.is_file()]
        if len(found) != 1:
            raise ValueError(f'Require exactly one 8-11 completion JSON for {condition}')
        value = json.loads(found[0].read_text())
        results.append(value)
    old.validate_results(results)
    return {r['condition']: (r, next(p for root in SOURCE_ROOTS
            if (p := root / 'results' / f'{r["condition"]}-fid50000.json').is_file())) for r in results}


def imported(row, source, path, args):
    value = protocol(row, SEEDS[0], args)
    if sampling_identity(source['protocol']) != sampling_identity(value):
        raise ValueError(f'8-11 sampling/code/model/reference mismatch: {row["condition"]}')
    return dict(row, seeds=SEEDS[0], num_images=50000, fid=source['fid'],
                protocol=value, protocol_id=old.digest(value),
                origin='8-11', source_path=str(path), source_result=source,
                generate_sec=0.0, fid_sec=0.0, generated_images=0,
                fid_ref=str(args.fid_ref), images='', gpu_name=source['gpu_name'])


def validate(results, definitions):
    seen = set()
    common, networks = None, {}
    for result in results:
        pair = key(result)
        if pair in seen:
            raise ValueError(f'Duplicate result: {pair}')
        seen.add(pair)
        definition = definitions.get(result['condition'])
        p = result['protocol']
        if definition is None or result['seeds'] not in SEEDS or result['num_images'] != 50000:
            raise ValueError('Unknown condition/seed/count')
        if not math.isfinite(float(result['fid'])) or float(result['fid']) < 0:
            raise ValueError('Invalid FID')
        if p.get('experiment') != '8-13' or result['protocol_id'] != old.digest(p):
            raise ValueError('Invalid protocol identity')
        expected = {k: v for k, v in definition.items() if k != 'network'}
        if p.get('condition') != expected or p.get('seeds') != result['seeds']:
            raise ValueError('Protocol condition/seed mismatch')
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError('Result condition differs from fixed configuration')
        for name, value in [('rho', 7), ('restart_noise', 1), ('sigma_min', 0.002),
                            ('sigma_max', 80), ('S_churn', 0), ('gen_batch', 64), ('fid_batch', 64)]:
            if p.get(name) != value:
                raise ValueError(f'Invalid protocol parameter {name}')
        for name in ('network', 'fid_ref', 'generate', 'fid', 'runner', 'base_runner'):
            ident = p.get(name, {})
            if set(ident) != {'sha256', 'size'} or not isinstance(ident['size'], int) or ident['size'] <= 0:
                raise ValueError(f'Invalid content identity {name}')
            if not isinstance(ident['sha256'], str) or len(ident['sha256']) != 64 or any(c not in '0123456789abcdef' for c in ident['sha256']):
                raise ValueError(f'Invalid SHA256 {name}')
        shared = {k: v for k, v in p.items() if k not in ('condition', 'network', 'seeds')}
        if common is not None and common != shared:
            raise ValueError('Mixed code/reference/batch protocols')
        common = shared
        cp = result['checkpoint']
        if cp in networks and networks[cp] != p['network']:
            raise ValueError('Mixed model identities')
        networks[cp] = p['network']
        if result.get('origin') == '8-11':
            source = result['source_result']
            old.validate_results([source])
            if (result['seeds'] != SEEDS[0] or source['fid'] != result['fid']
                    or sampling_identity(source['protocol']) != sampling_identity(p)):
                raise ValueError('Imported source/protocol mismatch')
        elif result.get('origin') != 'generated' or result['seeds'] == SEEDS[0]:
            raise ValueError('Invalid result origin')


def load_results(roots, rows):
    results = []
    for root in roots:
        if not (root / 'results').is_dir():
            continue
        for path in sorted((root / 'results').glob('*-fid50000-seeds-*.json')):
            row = json.loads(path.read_text())
            if path != marker(root, row):
                raise ValueError(f'Completion filename mismatch: {path}')
            results.append(row)
    validate(results, {r['condition']: r for r in rows})
    return results


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None,
                complete=len(values) == 3)


def aggregate(root, results, rows):
    validate(results, {r['condition']: r for r in rows})
    results = sorted(results, key=lambda r: (int(r['checkpoint'][:-1]), r['restart_k'], SEEDS.index(r['seeds'])))
    out = root / 'results'
    old.write_json(out / 'restart_fid_summary.json', results)
    fields = ['condition', 'checkpoint', 'restart_k', 'nfe', 'num_images', 'seeds', 'fid', 'origin', 'fid_ref', 'fid_sec']
    old.write_tsv(out / 'restart_fid_summary.tsv', [{k: r[k] for k in fields} for r in results], fields)
    lookup = {key(r): r for r in results}
    condition_stats, paired, improvements = [], [], []
    for row in rows:
        values = [lookup[(row['condition'], s)]['fid'] for s in SEEDS if (row['condition'], s) in lookup]
        condition_stats.append(dict(checkpoint=row['checkpoint'], restart_k=row['restart_k'],
                                    condition=row['condition'], **stats(values)))
    for cp, best in OPTIMAL.items():
        values = []
        for seeds in SEEDS:
            a, b = lookup.get((f'{cp}-k0', seeds)), lookup.get((f'{cp}-k{best}', seeds))
            if a is not None and b is not None:
                value = a['fid'] - b['fid']
                values.append(value)
                paired.append(dict(checkpoint=cp, optimal_k=best, seeds=seeds,
                                   baseline_fid=a['fid'], optimal_fid=b['fid'], improvement=value))
        improvements.append(dict(checkpoint=cp, optimal_k=best, positive_ranges=sum(v > 0 for v in values), **stats(values)))
    tables = [('condition_stats', condition_stats), ('paired_improvements', paired), ('improvement_stats', improvements)]
    for name, table in tables:
        old.write_json(out / f'{name}.json', table)
        if table:
            old.write_tsv(out / f'{name}.tsv', table, list(table[0]))
        elif (out / f'{name}.tsv').exists():
            (out / f'{name}.tsv').unlink()
    plot(out, lookup)


def plot(out, lookup):
    cache = Path(os.environ.setdefault('MPLCONFIGDIR', str(out.parent / '.cache/matplotlib')))
    cache.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xvalues = list(range(len(OPTIMAL)))
    for i, (cp, best) in enumerate(OPTIMAL.items()):
        for k, offset, color in [(0, -0.12, 'tab:gray'), (best, 0.12, 'tab:blue')]:
            values = [lookup[(f'{cp}-k{k}', s)]['fid'] for s in SEEDS if (f'{cp}-k{k}', s) in lookup]
            axes[0].scatter([i + offset] * len(values), values, color=color, alpha=.65)
            if len(values) == 3:
                axes[0].errorbar(i + offset, statistics.mean(values), yerr=statistics.stdev(values), color=color, fmt='s', capsize=4)
        diffs = [lookup[(f'{cp}-k0', s)]['fid'] - lookup[(f'{cp}-k{best}', s)]['fid'] for s in SEEDS
                 if (f'{cp}-k0', s) in lookup and (f'{cp}-k{best}', s) in lookup]
        axes[1].scatter([i] * len(diffs), diffs, color='tab:blue', alpha=.65)
        if len(diffs) == 3:
            axes[1].errorbar(i, statistics.mean(diffs), yerr=statistics.stdev(diffs), fmt='s', color='black', capsize=4)
    axes[0].set_ylabel('FID-50k')
    axes[0].set_title('Gray: K=0; blue: fixed optimal K')
    axes[1].set_ylabel('FID(K=0) - FID(optimal K)')
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set_title('Paired improvement; bars: mean +/- sample SD (n=3)')
    for axis in axes:
        axis.set_xticks(xvalues)
        axis.set_xticklabels(list(OPTIMAL))
        axis.set_xlabel('Training checkpoint (kimg)')
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'fid50k_3seeds.{suffix}')
    plt.close(fig)


def evaluate(row, seeds, args, pause):
    p = protocol(row, seeds, args)
    folder = args.exp_root / 'images' / row['condition'] / f'fid-n50000-seeds-{seeds}'
    old.ensure_protocol(folder, p)
    allowed = old.parse_seeds(seeds)
    log = args.exp_root / 'logs' / f'{row["condition"]}-seeds-{seeds}.log'
    pause.phase('validate', row, seeds, log)
    present = old.inventory(folder, allowed)
    missing = sorted(set(allowed) - set(present))
    generate_sec = 0
    for text in old.seed_ranges(missing):
        pause.phase('generate', log=log)
        elapsed, _ = old.run_logged(old.generate_command(row, folder, text, args), log)
        generate_sec += elapsed
    pause.phase('validate', log=log)
    old.inventory(folder, allowed, complete=True)
    pause.phase('fid', log=log)
    command = [args.torchrun, '--rdzv_backend=c10d', '--rdzv_endpoint=localhost:0',
               '--rdzv_id=experiment8-13', '--nproc_per_node=1', 'fid.py', 'calc',
               f'--images={folder}', f'--ref={args.fid_ref}', '--num=50000', '--batch=64']
    fid_sec, fid = old.run_logged(command, log)
    if fid is None or not math.isfinite(fid) or fid < 0:
        raise ValueError('Missing/nonfinite/negative FID')
    pause.phase('save', log=log)
    return dict(row, seeds=seeds, num_images=50000, fid=fid, protocol=p, protocol_id=old.digest(p),
                origin='generated', generate_sec=generate_sec, fid_sec=fid_sec,
                generated_images=len(missing), fid_ref=str(args.fid_ref), images=str(folder),
                gpu_name=os.environ['EXPECTED_GPU_NAME'])


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for option in ('plan-only', 'samples-only', 'verify-only', 'aggregate-only'):
        modes.add_argument(f'--{option}', action='store_true')
    parser.add_argument('--checkpoint-filter', default=os.environ.get('CHECKPOINT_FILTER', 'all'))
    parser.add_argument('--seed-filter', default=os.environ.get('SEED_FILTER', 'all'))
    parser.add_argument('--sample-seeds', default='0-1')
    parser.add_argument('--exp-root', type=Path, default=Path(os.environ.get('EXP_ROOT', str(ROOT))))
    parser.add_argument('--result-roots', type=Path, nargs='+')
    parser.add_argument('--fid-ref', type=Path, default=REPO / 'fid-refs/cifar10-32x32.npz')
    parser.add_argument('--torchrun', default=os.environ.get('TORCHRUN', str(Path(sys.executable).parent / 'torchrun')))
    args = parser.parse_args()
    args.exp_root = args.exp_root.resolve()
    args.fid_ref = args.fid_ref.resolve()
    args.gen_batch = args.fid_batch = 64
    return args


def main():
    args = arguments()
    all_rows = conditions()
    checkpoints = select(OPTIMAL, args.checkpoint_filter)
    rows = [r for r in all_rows if r['checkpoint'] in checkpoints]
    seeds = select(SEEDS, args.seed_filter)
    if args.plan_only:
        print(f'{len(rows) * len(seeds)} evaluations: {len(rows) if SEEDS[0] in seeds else 0} reused, '
              f'{len(rows) * len([s for s in seeds if s != SEEDS[0]])} new; '
              f'{len(rows) * len([s for s in seeds if s != SEEDS[0]]) * 50000} new images')
        for row in rows:
            for seed in seeds:
                print(row['condition'], seed, f'NFE={row["nfe"]}', 'reuse 8-11' if seed == SEEDS[0] else 'generate', row['network'])
        return
    if args.result_roots:
        for root in args.result_roots:
            if not (root / 'results').is_dir():
                raise ValueError(f'Missing result directory: {root / "results"}')
    results = load_results(args.result_roots or [args.exp_root], all_rows)
    if args.verify_only:
        definitions = {row['condition']: row for row in all_rows}
        for r in results:
            if r['checkpoint'] not in checkpoints or r['seeds'] not in seeds:
                continue
            if r['protocol'] != protocol(definitions[r['condition']], r['seeds'], args):
                raise ValueError('Completed result protocol changed')
            if r['origin'] == 'generated':
                # Only local output verification needs images; cross-PC aggregation does not.
                old.ensure_protocol(Path(r['images']), r['protocol'], create=False)
                old.inventory(Path(r['images']), old.parse_seeds(r['seeds']), complete=True)
        print(f'Verified {len(results)}/36 completed evaluations')
        return
    if args.aggregate_only:
        aggregate(args.exp_root, results, all_rows)
        print(f'Aggregated {len(results)}/36 evaluations')
        return
    if args.result_roots:
        raise ValueError('--result-roots is supported only with aggregate-only or verify-only')
    args.exp_root.mkdir(parents=True, exist_ok=True)
    with (args.exp_root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        legacy = legacy_results(rows)
        # Validate against the current files before importing or launching GPU work.
        for row in rows:
            imported(row, *legacy[row['condition']], args)
        lookup = {key(r): r for r in results}
        for r in results:
            definition = next(row for row in all_rows if row['condition'] == r['condition'])
            if r['protocol'] != protocol(definition, r['seeds'], args):
                raise ValueError('Completed result protocol changed')
        if not args.samples_only and SEEDS[0] in seeds:
            for row in rows:
                if (row['condition'], SEEDS[0]) not in lookup:
                    r = imported(row, *legacy[row['condition']], args)
                    old.write_json(marker(args.exp_root, r), r)
                    results.append(r)
                    lookup[key(r)] = r
        aggregate(args.exp_root, results, all_rows)
        jobs = [(r, s) for r in rows for s in seeds if (r['condition'], s) not in lookup and s != SEEDS[0]]
        if args.samples_only:
            jobs = [(r, args.sample_seeds) for r in rows]
            old.parse_seeds(args.sample_seeds)
        if not jobs:
            return
        pause = ConditionPause(args.exp_root, '8-13', args.exp_root / 'results/restart_fid_summary.tsv', args.fid_ref)
        terminal = 'error'
        try:
            for row, seed in jobs:
                pause.boundary(row, seed)
                if args.samples_only:
                    folder = args.exp_root / 'samples' / row['condition'] / f'seeds-{seed}'
                    old.ensure_protocol(folder, protocol(row, seed, args))
                    pause.phase('generate', row, seed, args.exp_root / 'logs/samples.log')
                    old.run_logged(old.generate_command(row, folder, seed, args), args.exp_root / 'logs/samples.log')
                else:
                    result = evaluate(row, seed, args, pause)
                    old.write_json(marker(args.exp_root, result), result)
                    results.append(result)
                    aggregate(args.exp_root, results, all_rows)
                    pause.phase('saved')
            terminal = 'complete'
        except StopRequested:
            terminal = 'exit'
        finally:
            pause.close(terminal)


if __name__ == '__main__':
    main()
