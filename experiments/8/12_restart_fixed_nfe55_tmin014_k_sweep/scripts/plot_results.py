#!/usr/bin/env python3
"""Redraw experiment 8-12 figures in the experiment 8-11 style, including partial results."""

import argparse
import json
import os
from pathlib import Path

import run_experiment as experiment


DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results"
COLOR_LIMIT = 0.4
MISSING_COLOR = "#eeeeee"


def load_groups(summary):
    results = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(results, list) or not results:
        raise ValueError("Summary must contain at least one completed result")
    experiment.validate_results(results)
    return experiment.comparison_rows(results)


def make_figure(groups, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    # An odd LUT length represents zero as pure white, as in experiment 8-11.
    colormap = LinearSegmentedColormap.from_list(
        "fid_delta", ["#3b4cc0", "#ffffff", "#b40426"], N=257)
    colormap.set_bad(MISSING_COLOR)
    norm = Normalize(vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, clip=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrix = np.full((len(experiment.CHECKPOINTS), len(experiment.KS)), np.nan)
    best_ks = np.full(len(experiment.CHECKPOINTS), np.nan)
    colors = plt.get_cmap("tab10")
    for i, checkpoint in enumerate(experiment.CHECKPOINTS):
        group = groups.get((checkpoint, count), {})
        if not group:
            continue
        # Keep each checkpoint's color even when earlier checkpoints are absent.
        ys = [group[k]["fid"] if k in group else np.nan for k in experiment.KS]
        axes[0].plot(experiment.KS, ys, marker="o", label=checkpoint, color=colors(i))
        if 0 in group:
            for j, k in enumerate(experiment.KS):
                if k in group:
                    matrix[i, j] = group[k]["fid"] - group[0]["fid"]
        if set(group) == set(experiment.KS):
            best_ks[i] = min(experiment.KS, key=lambda k: (group[k]["fid"], k))
    axes[0].set(xlabel="K", ylabel="FID", title=f"FID-{count}, NFE=55, t_min=0.14")
    axes[0].set_xticks(experiment.KS)
    if axes[0].lines:
        axes[0].legend(fontsize=8)

    heatmap = axes[1].imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=colormap, norm=norm)
    axes[1].set_xticks(range(len(experiment.KS)), experiment.KS)
    axes[1].set_yticks(range(len(experiment.CHECKPOINTS)), experiment.CHECKPOINTS)
    axes[1].set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
    below = bool(np.any(matrix < -COLOR_LIMIT))
    above = bool(np.any(matrix > COLOR_LIMIT))
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    figure.colorbar(heatmap, ax=axes[1], ticks=np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 9),
                   format=FormatStrFormatter("%.1f"), extend=extend)

    training = [int(checkpoint[:-1]) for checkpoint in experiment.CHECKPOINTS]
    # NaNs also prevent lines from bridging checkpoints with incomplete results.
    axes[2].plot(training, best_ks, marker="o")
    axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
    axes[2].set_xticks(training, experiment.CHECKPOINTS)
    axes[2].set_yticks(experiment.KS)
    axes[2].set_ylim(-0.4, 8.4)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_RESULTS / "restart_fid_summary.json",
                        help="summary JSON containing completed results; partial stages are supported")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS,
                        help="destination for PNG/SVG figures; result tables are never rewritten")
    args = parser.parse_args()
    groups = load_groups(args.summary)
    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RESULTS.parent / "logs" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for count in sorted({count for _, count in groups}):
        figure = make_figure(groups, count)
        try:
            for extension in ("png", "svg"):
                path = args.outdir / f"fixed_nfe55_tmin014_fid{count}.{extension}"
                temporary = path.with_name(f".{path.stem}.tmp.{extension}")
                figure.savefig(temporary, dpi=180)
                temporary.replace(path)
                print(path)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
