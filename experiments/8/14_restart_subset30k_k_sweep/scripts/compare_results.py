#!/usr/bin/env python3
"""Compare subset30k results with experiments 8-11 and 8-12 without modifying them."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCES = {'main18': '11_restart_fixed_main18_tmin014_k_sweep',
           'nfe55': '12_restart_fixed_nfe55_tmin014_k_sweep'}


def configured_checkpoints(mode):
    with (ROOT / 'config' / f'restart_{mode}.tsv').open() as handle:
        return sorted({row['checkpoint'] for row in csv.DictReader(handle, delimiter='\t')},
                      key=lambda value: int(value[:-1]))


def keyed(rows):
    result = {}
    for row in rows:
        key = row['checkpoint'], int(row['num_images']), int(row['restart_k'])
        if key in result:
            raise ValueError(f'Duplicate result: {key}')
        result[key] = row
    return result


def validate_pair(new, old):
    for key in ('checkpoint', 'main_steps', 'restart_steps', 'restart_k', 't_min', 't_max',
                'nfe', 'actual_t_min', 'restart_index', 'num_images', 'seeds', 'gen_batch', 'fid_batch'):
        if new[key] != old[key]:
            raise ValueError(f'Incompatible sampling/evaluation field: {key}')
    for key in ('fid_ref', 'generate', 'fid', 'rho', 'restart_noise', 'sigma_min', 'sigma_max', 'S_churn'):
        if new['protocol'][key] != old['protocol'][key]:
            raise ValueError(f'Incompatible reference/code/protocol field: {key}')


def build_comparison(subset, full):
    table = []
    for key, row in sorted(subset.items()):
        ck, count, k = key
        if key not in full or (ck, count, 0) not in subset:
            raise ValueError(f'Missing comparison/baseline: {key}')
        old = full[key]
        validate_pair(row, old)
        baseline = float(subset[ck, count, 0]['fid'])
        old_baseline = float(full[ck, count, 0]['fid'])
        table.append(dict(checkpoint=ck, num_images=count, restart_k=k,
                          subset_fid=float(row['fid']), full_fid=float(old['fid']),
                          subset_baseline=baseline, full_baseline=old_baseline,
                          subset_improvement=baseline-float(row['fid']),
                          full_improvement=old_baseline-float(old['fid'])))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'logs/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode, source in SOURCES.items():
        checkpoints = configured_checkpoints(mode)
        path = ROOT / 'runs' / mode / 'results/restart_fid_summary.json'
        subset = keyed(json.loads(path.read_text())) if path.exists() else {}
        if args.require_complete and set(subset) != {(ck, n, k) for ck in checkpoints
                                                    for n in (5000, 25000, 50000) for k in range(9)}:
            raise ValueError(f'{mode}: expected {len(checkpoints) * 27} unique evaluations')
        full = keyed(json.loads((REPO / 'experiments/8' / source / 'results/restart_fid_summary.json').read_text()))
        table = build_comparison(subset, full)
        if not table:
            continue
        with (out / f'{mode}_comparison.tsv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(table)
        for count in sorted({key[1] for key in subset}):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for i, ck in enumerate(checkpoints):
                selected = [r for r in table if r['checkpoint'] == ck and r['num_images'] == count]
                color = f'C{i}'
                for field, style in [('subset_fid', '-'), ('full_fid', '--')]:
                    axes[0].plot([r['restart_k'] for r in selected], [r[field] for r in selected],
                                 style, marker='o', color=color, label=f'{ck} {field.split("_")[0]}')
                axes[1].plot([r['restart_k'] for r in selected], [r['subset_improvement'] for r in selected],
                             marker='o', color=color, label=ck)
            for k in range(1, 9):
                old = [r for key, r in full.items() if key[1:] == (count, k)]
                xs = [float(full[r['checkpoint'], count, 0]['fid']) for r in old]
                ys = [x-float(r['fid']) for x, r in zip(xs, old)]
                color = plt.get_cmap('tab10')(k-1)
                axes[2].scatter(xs, ys, color=color, alpha=0.35, s=20)
                selected = [r for r in table if r['num_images'] == count and r['restart_k'] == k]
                axes[2].scatter([r['subset_baseline'] for r in selected], [r['subset_improvement'] for r in selected],
                                marker='*', s=100, color=color, label=f'K={k}')
                for r in selected:
                    axes[2].annotate(r['checkpoint'], (r['subset_baseline'], r['subset_improvement']), fontsize=6)
            axes[0].set(xlabel='K', ylabel='FID', title='Solid: subset30k; dashed: full50k')
            axes[1].set(xlabel='K', ylabel='FID(0) - FID(K)', title='Subset30k improvement')
            axes[2].set(xlabel='FID(K=0)', ylabel='FID(0) - FID(K)', title='Dots: full50k; stars: subset30k')
            for ax in axes:
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
            for ax in axes[1:]:
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            for ax in axes[:2]:
                ax.set_xticks(range(9))
            fig.suptitle(f'{mode}, FID-{count}, t_min=0.14')
            fig.tight_layout()
            for ext in ('png', 'svg'):
                fig.savefig(out / f'{mode}_comparison_fid{count}.{ext}', dpi=180)
            plt.close(fig)
        print(f'{mode}: compared {len(table)}/81 evaluations', flush=True)


if __name__ == '__main__':
    main()
