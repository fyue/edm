"""Snapshot the figures, result tables and progress for the 1024 progress report.

Runs the 3-seed comparison's own aggregation across both PCs' result roots (CPU only),
crops the experiment-number title from both K-sweep figures, and copies
everything the deck needs into assets/. Run from the EDM repository root with the edm
environment (Pillow required).
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

from PIL import Image

EXP_ROOT = 'experiments'
OUT_DIR = os.path.join(EXP_ROOT, '8', 'report', '1024', 'assets')

MAIN18 = os.path.join(EXP_ROOT, '8/11_restart_fixed_main18_tmin014_k_sweep')
NFE55 = os.path.join(EXP_ROOT, '8/12_restart_fixed_nfe55_tmin014_k_sweep')
SEEDS3 = os.path.join(EXP_ROOT, '8/13_restart_optimal_k_fid50k_3seeds')
POST200K = os.path.join(EXP_ROOT, '0/3_cifar10_cond_edm_post200k_extension')

TITLE_CROP = 150  # source pixels of the 'Experiment N-M' title above the subplot titles
RESULT_JSON = re.compile(r'^(\d+k)-k(\d)-fid(\d+)\.json$')


def copy(src, name):
    dst = os.path.join(OUT_DIR, name)
    shutil.copyfile(src, dst)
    print(f'{dst}: {src}')


def sweep_progress(json_dirs):
    done = {}
    for json_dir in json_dirs:
        for file in glob.glob(os.path.join(json_dir, '*.json')):
            match = RESULT_JSON.match(os.path.basename(file))
            if match:
                ckpt, k, stage = match.groups()
                done.setdefault(stage, set()).add(f'{ckpt}-k{k}')
    return {stage: len(conds) for stage, conds in done.items()}


def actual_tmin(json_dir_lists):
    table = {}
    for name, json_dirs in json_dir_lists.items():
        rows = []
        for k in range(9):
            files = [os.path.join(d, f'20k-k{k}-fid25000.json') for d in json_dirs]
            with open(next(f for f in files if os.path.exists(f))) as f:
                result = json.load(f)
            rows.append({key: result[key] for key in ('main_steps', 'nfe', 'actual_t_min', 'restart_index')})
        table[name] = rows
    return table


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3-seed comparison: combine both PCs' runs with the experiment's own aggregation.
    subprocess.run([sys.executable, os.path.join(SEEDS3, 'scripts/run_experiment.py'), '--aggregate-only',
                    '--result-roots', os.path.join(SEEDS3, 'runs/early'), os.path.join(SEEDS3, 'runs/late')], check=True)
    for name in ['fid50k_3seeds.png', 'condition_stats.tsv', 'improvement_stats.tsv', 'paired_improvements.tsv']:
        copy(os.path.join(SEEDS3, 'results', name), f'seeds3__{name}')

    for name, root, figure in [('fixed_main18', MAIN18, 'fixed_main18_tmin014_fid50000.png'),
                               ('fixed_nfe55', NFE55, 'fixed_nfe55_tmin014_fid50000.png')]:
        copy(os.path.join(root, 'results/fid_by_checkpoint_and_k.tsv'), f'{name}__fid_by_checkpoint_and_k.tsv')
        image = Image.open(os.path.join(root, 'results', figure))
        dst = os.path.join(OUT_DIR, f'{name}__fid50000.png')
        image.crop((0, TITLE_CROP, image.size[0], image.size[1])).save(dst)
        print(f'{dst}: cropped top {TITLE_CROP}px')

    for name in ['comparison.tsv', 'paired_improvements.tsv']:
        copy(os.path.join(POST200K, 'results', name), f'post200k__{name}')

    main18_dirs = [os.path.join(MAIN18, 'results'), os.path.join(MAIN18, 'runs/20k-140k/results')]
    nfe55_dirs = [os.path.join(NFE55, 'runs/20k-120k/results'), os.path.join(NFE55, 'runs/140k-200k/results')]
    seeds3_done = len([f for r in ('early', 'late') for f in glob.glob(os.path.join(SEEDS3, 'runs', r, 'results', '*-fid50000-seeds-*.json'))])
    snapshots = sorted(glob.glob(os.path.join(POST200K, 'training-runs/*/network-snapshot-*.pkl')))
    progress = {
        'fixed_main18': sweep_progress(main18_dirs),
        'fixed_nfe55': sweep_progress(nfe55_dirs),
        'seeds3_done': seeds3_done,
        'post200k_last_snapshot_kimg': int(re.search(r'(\d{6})\.pkl$', snapshots[-1]).group(1)),
    }
    with open(os.path.join(OUT_DIR, 'progress.json'), 'w') as f:
        json.dump(progress, f, indent=1)
    with open(os.path.join(OUT_DIR, 'actual_tmin.json'), 'w') as f:
        json.dump(actual_tmin({'fixed_main18': main18_dirs, 'fixed_nfe55': nfe55_dirs}), f, indent=1)
    print(progress)


if __name__ == '__main__':
    main()
