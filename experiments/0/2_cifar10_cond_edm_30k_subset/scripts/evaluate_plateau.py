#!/usr/bin/env python3
"""Evaluate FID early stopping from an experiment summary TSV."""

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--min-delta', type=Decimal, default=Decimal('0.1'))
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--through-kimg', type=int)
    parser.add_argument('--format', choices=['json', 'tsv'], default='json')
    return parser.parse_args()


def load_rows(path, through_kimg=None):
    expected_fields = ['kimg', 'seeds', 'num_images', 'fid', 'snapshot', 'images']
    try:
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if reader.fieldnames != expected_fields:
                raise ValueError(
                    f'expected TSV header {expected_fields}, got {reader.fieldnames}')
            rows = []
            seen_kimgs = set()
            for line_number, row in enumerate(reader, start=2):
                try:
                    kimg = int(row['kimg'])
                    fid = Decimal(row['fid'])
                except (ValueError, InvalidOperation) as err:
                    raise ValueError(f'invalid kimg or FID at line {line_number}') from err
                if kimg <= 0 or not math.isfinite(float(fid)):
                    raise ValueError(f'non-positive kimg or non-finite FID at line {line_number}')
                if kimg in seen_kimgs:
                    raise ValueError(f'duplicate kimg {kimg} at line {line_number}')
                seen_kimgs.add(kimg)
                if through_kimg is None or kimg <= through_kimg:
                    rows.append((kimg, fid))
    except OSError as err:
        raise ValueError(f'cannot read {path}: {err}') from err
    return sorted(rows)


def evaluate(rows, min_delta, patience):
    best_kimg = None
    best_fid = None
    stale_count = 0
    for kimg, fid in rows:
        if best_fid is None or fid <= best_fid - min_delta:
            best_kimg = kimg
            best_fid = fid
            stale_count = 0
        else:
            stale_count += 1
    should_stop = len(rows) >= patience + 1 and stale_count >= patience
    return {
        'evaluations': len(rows),
        'best_kimg': best_kimg,
        'best_fid': None if best_fid is None else str(best_fid),
        'stale_count': stale_count,
        'stop': should_stop,
        'stop_kimg': rows[-1][0] if should_stop else None,
    }


def main():
    args = parse_args()
    if args.min_delta < 0:
        raise SystemExit('--min-delta must be non-negative')
    if args.patience < 1:
        raise SystemExit('--patience must be positive')
    if args.through_kimg is not None and args.through_kimg <= 0:
        raise SystemExit('--through-kimg must be positive')
    try:
        result = evaluate(
            load_rows(args.summary, through_kimg=args.through_kimg),
            min_delta=args.min_delta,
            patience=args.patience,
        )
    except ValueError as err:
        raise SystemExit(str(err)) from err

    if args.format == 'tsv':
        values = [
            '1' if result['stop'] else '0',
            str(result['best_kimg'] or 'NA'),
            result['best_fid'] or 'NA',
            str(result['stale_count']),
            str(result['stop_kimg'] or 'NA'),
            str(result['evaluations']),
        ]
        print('\t'.join(values))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
