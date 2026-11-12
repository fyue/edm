#!/usr/bin/env python3
"""Recompute the RNG replay audit from stats.jsonl (CPU only).

`train.py --resume` does not restore the RNG position, so a fixed --seed returns
the noise and data order to the head of the stream at every launch. Two launches
on the same stream share their tick-to-tick loss fluctuations, so the correlation
of the detrended loss is close to 1. See ../rng_replay_audit.md for the results.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
MIN_TICKS = 25      # Shorter launches give no meaningful correlation.
RESTART_GAP = 5     # kimg. Tick 0 of a launch is one iteration past the previous tick.
WARMUP_TICKS = 40   # Ticks skipped when a launch starts from random initialization.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--window', type=int, default=10, help='ticks on each side of the moving average')
    return parser.parse_args()


def load_loss(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kimg = record.get('Progress/kimg', {}).get('mean')
            loss = record.get('Loss/loss', {}).get('mean')
            if kimg is not None and loss is not None:
                rows.append((kimg, loss))
    return rows


def split_launches(rows):
    launches, current = [], []
    for kimg, loss in rows:
        if current and (kimg <= current[-1][0] or kimg - current[-1][0] < RESTART_GAP):
            launches.append(current)
            current = []
        current.append((kimg, loss))
    if current:
        launches.append(current)
    return launches


def run_name(stats_path):
    name = str(stats_path.parent.relative_to(REPO / 'experiments')).replace('/training-runs', '')
    name = re.sub(r'^(\d)/(\d+)_[^/]*/', r'\1-\2 ', name)
    return re.sub(r'^1-cifar[^/]*/', '1 ', name)


def load_runs():
    paths = sorted((REPO / 'experiments').glob('**/training-runs/*/stats.jsonl'))
    return {run_name(path): split_launches(load_loss(path)) for path in paths}


def losses(launch, start=1, stop=None):
    return [loss for _, loss in launch][start:stop] # Tick 0 is a single iteration.


def detrend(values, window):
    return [value - statistics.mean(values[max(0, i - window):i + window + 1])
            for i, value in enumerate(values)]


def correlation(a, b):
    count = min(len(a), len(b))
    a, b = a[:count], b[:count]
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    scale = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / scale if scale else float('nan')


def find_run(runs, prefix):
    matches = [name for name in runs if name.startswith(prefix)]
    return runs[matches[0]] if matches else None


def main():
    args = parse_args()
    runs = load_runs()

    def series(launch, start=1, stop=None):
        return detrend(losses(launch, start, stop), args.window)

    run1 = find_run(runs, '1 ')
    if run1 is None:
        raise SystemExit('Experiment 1 stats.jsonl was not found; training-runs/ is not on this PC.')
    head_launch = next(launch for launch in run1 if launch[0][0] > 9000) # The launch from 10k.
    head = series(head_launch)

    print(f'# reference: experiment 1 launch from {head_launch[0][0]:.1f} kimg')
    print('run\tlaunch\tstart_kimg\tend_kimg\tticks\tcorr_vs_reference')
    for name, launches in runs.items():
        for index, launch in enumerate(launches):
            value = correlation(series(launch), head) if len(launch) > MIN_TICKS else float('nan')
            print(f'{name}\t{index}\t{launch[0][0]:.1f}\t{launch[-1][0]:.1f}\t{len(launch)}\t{value:+.3f}')

    print('\n# launches compared within their own lineage')
    subset30k = find_run(runs, '0-2 ')
    if subset30k:
        reference = series(subset30k[8]) # The launch from 50k.
        for launch in subset30k:
            if len(launch) > MIN_TICKS:
                print(f'0-2 from {launch[0][0]:.0f} vs 0-2 from 50k\t{correlation(series(launch), reference):+.3f}')
    pairs = [('0-3 ', 0, 1), ('0-5 ', 1, 2), ('7-2 ', 2, 3), ('3-2 ', 0, 1), ('3-6 ', 0, 1), ('3-7 ', 0, 1)]
    for prefix, left, right in pairs:
        launches = find_run(runs, prefix)
        if launches:
            print(f'{prefix.strip()} launch {left} vs {right}\t'
                  f'{correlation(series(launches[left]), series(launches[right])):+.3f}')

    print('\n# 0-3 against the head of 0-1, skipping the warmup')
    extension, retraining = find_run(runs, '0-3 '), find_run(runs, '0-1 ')
    if extension and retraining:
        for start in (WARMUP_TICKS, 100):
            value = correlation(series(extension[0], start, 201), series(retraining[0], start, 201))
            print(f'ticks {start}-201\t{value:+.3f}')

    print('\n# controls: uninterrupted runs should be close to 0')
    if retraining:
        for start in (10000, 50000, 100000, 150000):
            part = [row for row in retraining[0] if start < row[0] <= start + 10000]
            print(f'0-1 {start}-{start + 10000} vs reference\t{correlation(series(part, 0), head):+.3f}')
    continuous = find_run(runs, '0-6 ')
    if continuous and subset30k:
        part = [row for row in continuous[0] if row[0] > 10000]
        if len(part) > MIN_TICKS:
            print(f'0-6 past 10k ({len(part)} ticks) vs 0-2 from 50k\t'
                  f'{correlation(series(part, 0), series(subset30k[8])):+.3f}')


if __name__ == '__main__':
    main()
