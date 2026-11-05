#!/usr/bin/env python3
"""Detect replayed training streams from stats.jsonl (CPU only).

The loss of one tick depends on the sampled images and noise. If two segments
use the same random stream, the small tick-to-tick fluctuations of their losses
line up, so the correlation of the detrended loss is close to 1. Independent
streams give a value close to 0.
"""

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_STATS = (Path(__file__).resolve().parents[1] / 'training-runs'
                 / 'cifar10-cond-ddpmpp-edm-seed0-subset30k-continuous' / 'stats.jsonl')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stats', type=Path, default=DEFAULT_STATS)
    parser.add_argument('--segment-kimg', type=int, default=10000)
    parser.add_argument('--window', type=int, default=10, help='ticks on each side of the moving average')
    parser.add_argument('--min-ticks', type=int, default=30)
    return parser.parse_args()


def load_loss(path):
    rows = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            kimg = record.get('Progress/kimg', {}).get('mean')
            loss = record.get('Loss/loss', {}).get('mean')
            if kimg is not None and loss is not None:
                rows[kimg] = loss # A later launch overwrites a repeated tick.
    return sorted(rows.items())


def detrend(values, window):
    return [value - statistics.mean(values[max(0, i - window):i + window + 1])
            for i, value in enumerate(values)]


def correlation(a, b):
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    scale = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / scale if scale else float('nan')


def main():
    args = parse_args()
    segments = {}
    for kimg, loss in load_loss(args.stats):
        # Tick kimg values sit just past a boundary, so index by the segment end.
        segments.setdefault(int((kimg - 1e-6) // args.segment_kimg), []).append(loss)
    print('segment_a_kimg\tsegment_b_kimg\tticks\tcorrelation')
    indices = sorted(segments)
    for left, right in zip(indices, indices[1:]):
        count = min(len(segments[left]), len(segments[right]))
        if left == 0 or right != left + 1 or count < args.min_ticks:
            continue # The first segment is dominated by the warmup trend.
        value = correlation(detrend(segments[left][:count], args.window),
                            detrend(segments[right][:count], args.window))
        print(f'{left * args.segment_kimg}\t{right * args.segment_kimg}\t{count}\t{value:.3f}')


if __name__ == '__main__':
    main()
