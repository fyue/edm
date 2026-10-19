#!/usr/bin/env python3
"""Redraw the merged experiment 8-11 figures with a white, centered zero."""

import argparse
import json
import os
from pathlib import Path

import run_experiment as experiment


DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results"
COLOR_LIMIT = 0.4


def load_groups(summary):
    results = json.loads(summary.read_text(encoding="utf-8"))
    experiment.validate_results(results)
    expected = {(checkpoint, count, k) for checkpoint in experiment.CHECKPOINTS
                for count in experiment.STAGES for k in experiment.KS}
    actual = {(row["checkpoint"], row["num_images"], row["restart_k"]) for row in results}
    if actual != expected or len(results) != len(expected):
        raise ValueError("Plotting requires all 270 merged experiment 8-11 results")
    groups = experiment.comparison_rows(results)
    for (checkpoint, count), group in groups.items():
        for k, row in group.items():
            delta = row["fid"] - group[0]["fid"]
            if abs(delta) > COLOR_LIMIT:
                raise ValueError(f"{checkpoint}, FID-{count}, K={k}: delta {delta} exceeds +/-{COLOR_LIMIT}")
    return groups


def make_figure(groups, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    # An odd LUT length includes pure white exactly at normalized zero (0.5).
    colormap = LinearSegmentedColormap.from_list(
        "fid_delta", ["#3b4cc0", "#ffffff", "#b40426"], N=257)
    norm = Normalize(vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrix = np.empty((len(experiment.CHECKPOINTS), len(experiment.KS)))
    best_ks = []
    for i, checkpoint in enumerate(experiment.CHECKPOINTS):
        group = groups[checkpoint, count]
        xs = sorted(group)
        axes[0].plot(xs, [group[k]["fid"] for k in xs], marker="o", label=checkpoint)
        matrix[i] = [group[k]["fid"] - group[0]["fid"] for k in experiment.KS]
        best_ks.append(min(experiment.KS, key=lambda k: (group[k]["fid"], k)))
    axes[0].set(xlabel="K", ylabel="FID", title=f"FID-{count}, main=18, t_min=0.14")
    axes[0].set_xticks(experiment.KS)
    axes[0].legend(fontsize=8)
    heatmap = axes[1].imshow(matrix, aspect="auto", cmap=colormap, norm=norm)
    axes[1].set_xticks(range(len(experiment.KS)), experiment.KS)
    axes[1].set_yticks(range(len(experiment.CHECKPOINTS)), experiment.CHECKPOINTS)
    axes[1].set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
    figure.colorbar(heatmap, ax=axes[1], ticks=np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 9),
                   format=FormatStrFormatter("%.1f"))
    axes[2].plot([int(checkpoint[:-1]) for checkpoint in experiment.CHECKPOINTS], best_ks, marker="o")
    axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
    axes[2].set_xticks([int(checkpoint[:-1]) for checkpoint in experiment.CHECKPOINTS],
                      experiment.CHECKPOINTS)
    axes[2].set_yticks(experiment.KS)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_RESULTS / "restart_fid_summary.json",
                        help="merged summary JSON containing all 270 results")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS,
                        help="destination for PNG/SVG figures; result tables are never rewritten")
    args = parser.parse_args()
    groups = load_groups(args.summary)
    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RESULTS.parent / "logs" / "matplotlib"))
    for count in experiment.STAGES:
        figure = make_figure(groups, count)
        for extension in ("png", "svg"):
            path = args.outdir / f"fixed_main18_tmin014_fid{count}.{extension}"
            figure.savefig(path, dpi=180)
            print(path)
        import matplotlib.pyplot as plt
        plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
