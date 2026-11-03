#!/usr/bin/env python3
"""Build CPU-only figures from the TSV snapshots beside this script."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / 'data' / name).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write(name, rows):
    with (ROOT / 'data' / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def save(fig, stem):
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ('png', 'svg'):
        fig.savefig(ROOT / 'figures' / f'{stem}.{ext}', dpi=170)
    plt.close(fig)


def groups(name):
    result = {}
    for row in read(name):
        if int(row['num_images']) != 50000:
            continue
        if row['seeds'] != '0-49999':
            raise ValueError('Expected the same seed set 0-49999')
        key = int(row['checkpoint'][:-1])
        k = int(row['restart_k'])
        group = result.setdefault(key, {})
        if k in group:
            raise ValueError('Duplicate checkpoint/K')
        group[k] = float(row['fid'])
    return result


def complete(group):
    return set(group) == set(range(9))


def sweep(name, stem, label):
    data = groups(name)
    steps = sorted(data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(label + ': FID-50k (one generation seed range)')
    baseline, minimum, means, stats = [], [], [], []
    matrix = np.full((len(steps), 9), np.nan)
    for i, step in enumerate(steps):
        group = data[step]
        baseline.append(group.get(0, np.nan))
        if not complete(group):
            minimum.append(np.nan)
            means.append(np.nan)
            continue
        best = min(group, key=lambda k: (group[k], k))
        low = group[best]
        near = [k for k in range(9) if group[k] - low <= 0.02 + 1e-12]
        mean = float(np.mean(near))
        minimum.append(low)
        means.append(mean)
        matrix[i] = [group[k] - group[0] for k in range(9)]
        for k in range(1, 9):
            axes[1, 0].scatter(group[0], group[0] - group[k], color=plt.cm.viridis(k / 8), s=24)
        axes[1, 0].annotate(f'{step}k', (group[0], group[0] - low), fontsize=8)
        axes[1, 1].scatter([step] * len(near), near, color='#9ecae1', s=48)
        axes[1, 1].scatter(step, best, color='black', s=36, zorder=4)
        stats.append(dict(checkpoint=f'{step}k', baseline_fid=group[0], best_k=best,
                          min_fid=low, improvement=group[0] - low,
                          near_optimal_ks=','.join(map(str, near)), mean_k=mean,
                          fid_gap_threshold=0.02))
    axes[0, 0].plot(steps, baseline, '-o', label='K=0')
    axes[0, 0].plot(steps, minimum, '-o', label='Minimum over K=0..8')
    axes[0, 0].set(xlabel='Training (kimg)', ylabel='FID', title='Baseline and best evaluated Restart')
    axes[0, 0].legend()
    vmax = max(float(np.nanmax(np.abs(matrix))), 0.001)
    image = axes[0, 1].imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_xticks(range(9), range(9))
    axes[0, 1].set_yticks(range(len(steps)), [f'{s}k' for s in steps])
    axes[0, 1].set(xlabel='K', ylabel='Training (kimg)', title='FID(K) - FID(0): blue = better')
    fig.colorbar(image, ax=axes[0, 1])
    axes[1, 0].axhline(0, color='grey', linewidth=1)
    axes[1, 0].set(xlabel='Baseline FID(K=0)', ylabel='FID(0) - FID(K)', title='Improvement for each K=1..8')
    axes[1, 1].plot(steps, means, '-o', color='#ff7f0e', label='Arithmetic mean of near-optimal K')
    axes[1, 1].set(xlabel='Training (kimg)', ylabel='K', title='Best K (black), near-optimal K (light blue)')
    axes[1, 1].set_yticks(range(9))
    axes[1, 1].legend(fontsize=8)
    fig.text(0.5, 0.012, 'Near-optimal: FID(K) - min FID <= 0.02. Mean K is a descriptive center, not an optimum or significance test.', ha='center', fontsize=9)
    save(fig, stem)
    write(stem + '_stats.tsv', stats)


def training():
    rows = sorted([row for row in read('subset30k_training_fid.tsv') if row['num_images'] == '50000'], key=lambda row: int(row['kimg']))
    if len({row['kimg'] for row in rows}) != len(rows):
        raise ValueError('Ambiguous training FID rows')
    fig, axis = plt.subplots(figsize=(9, 5))
    xs, ys = [int(r['kimg']) / 1000 for r in rows], [float(r['fid']) for r in rows]
    axis.plot(xs, ys, '-o')
    for x, y in zip(xs, ys):
        axis.annotate(f'{y:.3f}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    best = min(range(len(ys)), key=lambda i: ys[i])
    axis.scatter(xs[best], ys[best], color='red', s=60, zorder=4, label=f'Best observed: {xs[best]:g}k')
    axis.set(xlabel='Training (kimg)', ylabel='FID-50k', title='30,000 training images: baseline EDM trajectory')
    axis.legend()
    fig.text(0.5, 0.015, 'Single training run and generation seed range. FID worsening suggests overfitting; this plot alone does not prove it.', ha='center', fontsize=8)
    save(fig, '05_subset30k_training')


def reported_three_seeds():
    for source, stem, title in [
        ('slack_reported_full_3seeds.tsv', '06_full_3seeds', 'Full-data main=18: K=0 vs selected K'),
        ('slack_reported_post200k_3seeds.tsv', '07_post200k_3seeds', 'Post-200k main=18: K=0 vs K=1'),
    ]:
        rows = read(source)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.suptitle(title + ' (reported in Slack, 2026-10-27)')
        steps = [int(r['checkpoint'][:-1]) for r in rows]
        for key, offset, label in [('baseline', -1.5, 'K=0'), ('restart', 1.5, 'Selected Restart K')]:
            axes[0].errorbar(np.array(steps) + offset, [float(r[key + '_mean']) for r in rows],
                             yerr=[float(r[key + '_std']) for r in rows], marker='o', capsize=4, label=label)
        axes[0].set(xlabel='Training (kimg)', ylabel='FID-50k, mean +/- sample SD', title='3 generation seed ranges, fixed training run')
        axes[0].legend()
        improvements = [float(r['improvement_mean']) for r in rows]
        errors = [float(r['improvement_std']) for r in rows] if 'improvement_std' in rows[0] else None
        axes[1].errorbar(steps, improvements, yerr=errors, marker='o', capsize=4)
        axes[1].axhline(0, color='grey', linewidth=1)
        axes[1].set(xlabel='Training (kimg)', ylabel='FID(0) - FID(Restart)', title='Paired improvement' if errors else 'Mean improvement (SD not reported)')
        fig.text(0.5, 0.012, 'Transcribed reported summary values; not recomputed from the incomplete local seed-level snapshot. SD is not a confidence interval.', ha='center', fontsize=8)
        save(fig, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    (ROOT / 'figures').mkdir(exist_ok=True)
    for source, stem, label in [
        ('full_main18.tsv', '01_full_main18', '50k training images, main=18, t_min=0.14'),
        ('full_nfe55.tsv', '02_full_nfe55', '50k training images, NFE=55, nominal t_min=0.14'),
        ('subset_main18.tsv', '03_subset_main18', '30k training images, main=18, t_min=0.14'),
        ('subset_nfe55.tsv', '04_subset_nfe55', '30k training images, NFE=55, nominal t_min=0.14'),
    ]:
        sweep(source, stem, label)
    training()
    reported_three_seeds()
    actual = {}
    for row in read('full_nfe55.tsv'):
        k = int(row['restart_k'])
        actual[k] = dict(k=k, main_steps=row['main_steps'], nfe=row['nfe'],
                         nominal_t_min=row['t_min'], actual_t_min=row['actual_t_min'], restart_index=row['restart_index'])
    write('actual_tmin_nfe55.tsv', [actual[k] for k in sorted(actual)])
    print('Built 7 PNG/SVG figures and calculation tables; figure 08 is a copied local t_min comparison.')


if __name__ == '__main__':
    main()
