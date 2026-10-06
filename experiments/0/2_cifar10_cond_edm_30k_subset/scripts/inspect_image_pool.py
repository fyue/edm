#!/usr/bin/env python3
"""Inspect a subdirectory-based generated image pool and list missing seeds."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--format', choices=['json', 'tsv'], default='json')
    return parser.parse_args()


def compact_ranges(values):
    ranges = []
    if not values:
        return ranges
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f'{start}-{previous}')
        start = previous = value
    ranges.append(str(start) if start == previous else f'{start}-{previous}')
    return ranges


def inspect_pool(pool, count):
    if count < 1:
        raise ValueError('--count must be positive')
    if pool.exists() and not pool.is_dir():
        raise ValueError(f'image pool is not a directory: {pool}')

    seen = set()
    errors = []
    if pool.is_dir():
        for path in pool.rglob('*.png'):
            try:
                seed = int(path.stem)
            except ValueError:
                errors.append(f'invalid PNG filename: {path}')
                continue
            expected_relative = Path(f'{seed - seed % 1000:06d}') / f'{seed:06d}.png'
            if path.relative_to(pool) != expected_relative:
                errors.append(f'unexpected PNG path: {path}')
                continue
            if not 0 <= seed < count:
                errors.append(f'unexpected seed {seed}: {path}')
                continue
            if path.stat().st_size == 0:
                errors.append(f'empty PNG file: {path}')
                continue
            if seed in seen:
                errors.append(f'duplicate seed {seed}: {path}')
                continue
            seen.add(seed)
    if errors:
        preview = '; '.join(errors[:10])
        if len(errors) > 10:
            preview += f'; ... ({len(errors)} errors)'
        raise ValueError(preview)

    missing = sorted(set(range(count)) - seen)
    return {
        'actual_count': len(seen),
        'missing_count': len(missing),
        'missing_seeds': ','.join(compact_ranges(missing)),
        'complete': not missing,
    }


def main():
    args = parse_args()
    try:
        result = inspect_pool(args.pool, args.count)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    if args.format == 'tsv':
        print('\t'.join([
            str(result['actual_count']),
            str(result['missing_count']),
            result['missing_seeds'] or 'NA',
            '1' if result['complete'] else '0',
        ]))
    else:
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
