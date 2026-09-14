#!/usr/bin/env python
"""Summarize Restart improvement against same-main and same-NFE ODE baselines."""

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path('experiments/8/1_restart_sampling_checkpoint_sweep/results/restart_fid_summary.tsv')
DEFAULT_OUTPUT = Path('experiments/8/1_restart_sampling_checkpoint_sweep/results/restart_improvements.tsv')


def protocol_key(row):
    return row['checkpoint'], row['num_images'], row['seeds'], row['fid_ref']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))

    same_main = {}
    same_nfe = {}
    for row in rows:
        if row['method'] != 'ode-heun':
            continue
        key = protocol_key(row)
        same_nfe[key + (row['nfe'],)] = float(row['fid'])
        if row['main_steps'] == '18':
            same_main[key] = float(row['fid'])

    output_rows = []
    for row in rows:
        if row['method'] != 'restart':
            continue
        key = protocol_key(row)
        main_fid = same_main.get(key)
        nfe_fid = same_nfe.get(key + (row['nfe'],))
        restart_fid = float(row['fid'])
        output_rows.append({
            'condition': row['condition'],
            'checkpoint': row['checkpoint'],
            'restart_k': row['restart_k'],
            'nfe': row['nfe'],
            'num_images': row['num_images'],
            'seeds': row['seeds'],
            'restart_fid': f'{restart_fid:.8g}',
            'same_main_ode_fid': 'NA' if main_fid is None else f'{main_fid:.8g}',
            'improvement_vs_same_main': 'NA' if main_fid is None else f'{main_fid - restart_fid:.8g}',
            'same_nfe_ode_fid': 'NA' if nfe_fid is None else f'{nfe_fid:.8g}',
            'improvement_vs_same_nfe': 'NA' if nfe_fid is None else f'{nfe_fid - restart_fid:.8g}',
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0]) if output_rows else [
        'condition', 'checkpoint', 'restart_k', 'nfe', 'num_images', 'seeds',
        'restart_fid', 'same_main_ode_fid', 'improvement_vs_same_main',
        'same_nfe_ode_fid', 'improvement_vs_same_nfe',
    ]
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter='\t', lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f'Wrote {len(output_rows)} Restart comparisons to {args.output}')


if __name__ == '__main__':
    main()
