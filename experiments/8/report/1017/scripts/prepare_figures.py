"""Snapshot the figures, result tables and progress for the 1017 progress report.

The K-sweep runs rewrite their figures and tables while they are still evaluating, so
the deck is built from copies taken at one moment. Only this PC's run (20k-120k) of the
fixed-NFE sweep is used. Run from the EDM repository root (Pillow not required).
"""

import glob
import json
import os
import re
import shutil
import sys

EXP_ROOT = 'experiments/8'
OUT_DIR = os.path.join(EXP_ROOT, 'report', '1017', 'assets')

SWEEPS = {
    # Fixed main path (18 steps); NFE grows with K.
    'fixed_main18': {
        'results': os.path.join(EXP_ROOT, '11_restart_fixed_main18_tmin014_k_sweep/results'),
        'json_dirs': [
            os.path.join(EXP_ROOT, '11_restart_fixed_main18_tmin014_k_sweep/results'),
            os.path.join(EXP_ROOT, '11_restart_fixed_main18_tmin014_k_sweep/runs/20k-140k/results'),
        ],
        'figures': ['fixed_main18_tmin014_fid50000.png'],
    },
    # Fixed compute (55 NFE); this PC's checkpoints only.
    'fixed_nfe55': {
        'results': os.path.join(EXP_ROOT, '12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k/results'),
        'json_dirs': [os.path.join(EXP_ROOT, '12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k/results')],
        'figures': ['fixed_nfe55_tmin014_fid25000.png'],
    },
}
RESULT_JSON = re.compile(r'^(\d+k)-k(\d)-fid(\d+)\.json$')


def completed_conditions(json_dirs):
    done = {}
    for json_dir in json_dirs:
        for file in glob.glob(os.path.join(json_dir, '*.json')):
            match = RESULT_JSON.match(os.path.basename(file))
            if match:
                ckpt, k, stage = match.groups()
                done.setdefault(stage, set()).add(f'{ckpt}-k{k}')
    return {stage: sorted(conds) for stage, conds in done.items()}


def write_actual_tmin():
    """Record main steps, NFE and the actual Restart start sigma per K (from 20k, FID-25k)."""
    table = {}
    for name, sweep in SWEEPS.items():
        rows = []
        for k in range(9):
            files = [os.path.join(d, f'20k-k{k}-fid25000.json') for d in sweep['json_dirs']]
            with open(next(f for f in files if os.path.exists(f))) as f:
                result = json.load(f)
            rows.append({key: result[key] for key in ('main_steps', 'nfe', 'actual_t_min', 'restart_index')})
        table[name] = rows
    dst = os.path.join(OUT_DIR, 'actual_tmin.json')
    with open(dst, 'w') as f:
        json.dump(table, f, indent=1)
    print(f'{dst}: written')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    write_actual_tmin()
    if '--tmin-only' in sys.argv[1:]:
        return
    progress = {}
    for name, sweep in SWEEPS.items():
        for file in sweep['figures'] + ['fid_by_checkpoint_and_k.tsv']:
            dst = os.path.join(OUT_DIR, f'{name}__{file}')
            shutil.copyfile(os.path.join(sweep['results'], file), dst)
            print(f'{dst}: copied')
        progress[name] = completed_conditions(sweep['json_dirs'])
    dst = os.path.join(OUT_DIR, 'progress.json')
    with open(dst, 'w') as f:
        json.dump(progress, f, indent=1)
    for name, stages in progress.items():
        print(name, {stage: len(conds) for stage, conds in sorted(stages.items(), key=lambda s: int(s[0]))})


if __name__ == '__main__':
    main()
