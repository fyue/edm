"""Snapshot the figures, result tables and progress for the 1107 progress report.

The two 3-seed checks keep their results in more than one root, so they are aggregated here with
each experiment's own CPU-only aggregation. The output goes to a temporary root and only the
needed files are copied into assets/; the experiments' results/ directories are left as they are.
Run from the EDM repository root with the edm environment (Pillow required).
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

EXP_ROOT = 'experiments'
OUT_DIR = os.path.join(EXP_ROOT, '8', 'report', '1107', 'assets')

MAIN18 = os.path.join(EXP_ROOT, '8/11_restart_fixed_main18_tmin014_k_sweep')
NFE55 = os.path.join(EXP_ROOT, '8/12_restart_fixed_nfe55_tmin014_k_sweep')
SEEDS3 = os.path.join(EXP_ROOT, '8/13_restart_optimal_k_fid50k_3seeds')
SUBSET_SWEEP = os.path.join(EXP_ROOT, '8/14_restart_subset30k_k_sweep')
POST200K = os.path.join(EXP_ROOT, '0/3_cifar10_cond_edm_post200k_extension')
POST200K_SEEDS3 = os.path.join(EXP_ROOT, '0/4_post200k_restart_k1_fid50k_3seeds')
SUBSET = os.path.join(EXP_ROOT, '0/2_cifar10_cond_edm_30k_subset')

TITLE_CROP = 150  # source pixels of the 'Experiment N-M' title above the subplot titles
RESULT_JSON = re.compile(r'^(\d+k)-k(\d)-fid(\d+)\.json$')


def copy(src, name):
    dst = os.path.join(OUT_DIR, name)
    shutil.copyfile(src, dst)
    print(f'{dst}: {src}')


def aggregate(experiment, roots, prefix, names):
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, 'results'))
        subprocess.run([sys.executable, os.path.join(experiment, 'scripts/run_experiment.py'), '--aggregate-only',
                        '--exp-root', root, '--result-roots', *roots], check=True)
        for name in names:
            copy(os.path.join(root, 'results', name), f'{prefix}__{name}')


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


def draw_subset_sweep(table_path, output):
    """Rebuild the four subset panels directly from the saved experiment table."""
    import math

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    groups = {}
    with open(table_path, encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if int(row['num_images']) != 50000:
                continue
            step, k, fid = int(row['checkpoint'][:-1]), int(row['restart_k']), float(row['fid'])
            group = groups.setdefault(step, {})
            if k not in range(9) or k in group or not math.isfinite(fid) or fid < 0:
                raise ValueError('Invalid or duplicate subset FID-50k result')
            group[k] = fid
    if not groups or not any(set(group) == set(range(9)) for group in groups.values()):
        raise ValueError('No complete subset K=0..8 sweep')

    steps = sorted(groups)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    baseline, minimum, means = [], [], []
    matrix = np.full((len(steps), 9), np.nan)
    for i, step in enumerate(steps):
        group = groups[step]
        baseline.append(group.get(0, np.nan))
        if set(group) != set(range(9)):
            minimum.append(np.nan)
            means.append(np.nan)
            continue
        best = min(group, key=lambda k: (group[k], k))
        low = group[best]
        near = [k for k in range(9) if group[k] - low <= 0.02 + 1e-12]
        minimum.append(low)
        means.append(float(np.mean(near)))
        matrix[i] = [group[k] - group[0] for k in range(9)]
        for k in range(1, 9):
            axes[1, 0].scatter(group[0], group[0] - group[k], color=plt.cm.viridis(k / 8), s=24)
        axes[1, 0].annotate(f'{step}k', (group[0], group[0] - low), fontsize=8)
        axes[1, 1].scatter([step] * len(near), near, color='#9ecae1', s=48)
        axes[1, 1].scatter(step, best, color='black', s=36, zorder=4)
    axes[0, 0].plot(steps, baseline, '-o', label='K=0')
    axes[0, 0].plot(steps, minimum, '-o', label='Minimum over K=0..8')
    axes[0, 0].set(xlabel='Training (kimg)', ylabel='FID', title='Baseline and best evaluated Restart')
    axes[0, 0].legend()
    limit = max(float(np.nanmax(np.abs(matrix))), 0.001)
    image = axes[0, 1].imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap='bwr',
                            vmin=-limit, vmax=limit)
    axes[0, 1].set_xticks(range(9), range(9))
    axes[0, 1].set_yticks(range(len(steps)), [f'{s}k' for s in steps])
    axes[0, 1].set(xlabel='K', ylabel='Training (kimg)', title='FID(K) - FID(0): blue = better')
    figure.colorbar(image, ax=axes[0, 1])
    axes[1, 0].axhline(0, color='grey', linewidth=1)
    axes[1, 0].set(xlabel='Baseline FID(K=0)', ylabel='FID(0) - FID(K)', title='Improvement for each K=1..8')
    axes[1, 1].plot(steps, means, '-o', color='#ff7f0e', label='Arithmetic mean of near-optimal K')
    axes[1, 1].set(xlabel='Training (kimg)', ylabel='K', title='Best K (black), near-optimal K (light blue)')
    axes[1, 1].set_yticks(range(9))
    axes[1, 1].legend(fontsize=8)
    figure.tight_layout()
    try:
        figure.savefig(output, dpi=170)
    finally:
        plt.close(figure)


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=OUT_DIR, help='destination for report assets')
    OUT_DIR = parser.parse_args().outdir
    os.makedirs(OUT_DIR, exist_ok=True)

    aggregate(SEEDS3, [os.path.join(SEEDS3, 'runs/early'), os.path.join(SEEDS3, 'runs/late')], 'seeds3',
              ['fid50k_3seeds.png', 'condition_stats.tsv', 'improvement_stats.tsv', 'paired_improvements.tsv'])
    aggregate(POST200K_SEEDS3, [POST200K_SEEDS3, os.path.join(POST200K_SEEDS3, 'runs/pascal')], 'post200k_seeds3',
              ['condition_stats.tsv', 'improvement_stats.tsv', 'paired_improvements.tsv'])

    for name, root, figure in [('fixed_main18', MAIN18, 'fixed_main18_tmin014_fid50000_mean_k.png'),
                               ('fixed_nfe55', NFE55, 'fixed_nfe55_tmin014_fid50000_mean_k.png')]:
        copy(os.path.join(root, 'results/fid_by_checkpoint_and_k.tsv'), f'{name}__fid_by_checkpoint_and_k.tsv')
        image = Image.open(os.path.join(root, 'results', figure))
        dst = os.path.join(OUT_DIR, f'{name}__fid50000_mean_k.png')
        image.crop((0, TITLE_CROP, image.size[0], image.size[1])).save(dst)
        print(f'{dst}: cropped top {TITLE_CROP}px')

    # Draw from the copied table so the figure and its saved source always agree.
    for name in ('main18', 'nfe55'):
        copy(os.path.join(SUBSET_SWEEP, 'runs', name, 'results/fid_by_checkpoint_and_k.tsv'),
             f'subset_{name}__fid_by_checkpoint_and_k.tsv')
        dst = os.path.join(OUT_DIR, f'subset_{name}__fid50000.png')
        draw_subset_sweep(os.path.join(OUT_DIR, f'subset_{name}__fid_by_checkpoint_and_k.tsv'), dst)
        print(f'{dst}: redrawn from saved experiment table')

    for name in ['comparison.tsv', 'paired_improvements.tsv']:
        copy(os.path.join(POST200K, 'results', name), f'post200k__{name}')
    copy(os.path.join(SUBSET, 'results/fid_summary.tsv'), 'subset__fid_summary.tsv')

    main18_dirs = [os.path.join(MAIN18, 'results'), os.path.join(MAIN18, 'runs/20k-140k/results')]
    nfe55_dirs = [os.path.join(NFE55, 'runs/20k-120k/results'), os.path.join(NFE55, 'runs/140k-200k/results')]
    snapshots = sorted(glob.glob(os.path.join(POST200K, 'training-runs/*/network-snapshot-*.pkl')))
    progress = {
        'fixed_main18': sweep_progress(main18_dirs),
        'fixed_nfe55': sweep_progress(nfe55_dirs),
        'subset_main18': sweep_progress([os.path.join(SUBSET_SWEEP, 'runs/main18/results')]),
        'subset_nfe55': sweep_progress([os.path.join(SUBSET_SWEEP, 'runs/nfe55/results')]),
        'post200k_last_snapshot_kimg': int(re.search(r'(\d{6})\.pkl$', snapshots[-1]).group(1)),
    }
    with open(os.path.join(OUT_DIR, 'progress.json'), 'w') as f:
        json.dump(progress, f, indent=1)
    with open(os.path.join(OUT_DIR, 'actual_tmin.json'), 'w') as f:
        json.dump(actual_tmin({'fixed_main18': main18_dirs, 'fixed_nfe55': nfe55_dirs}), f, indent=1)
    print(progress)


if __name__ == '__main__':
    main()
