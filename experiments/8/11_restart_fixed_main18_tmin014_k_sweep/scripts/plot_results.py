#!/usr/bin/env python3
"""Draw experiment 8-11 overview and baseline-FID versus improvement figures."""

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


def make_figure(groups, count, near_optimal=False, mean_k=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    figure.suptitle(f"Experiment 8-11: FID-{count:,}, main=18, t_min=0.14", fontsize=14)
    matrix = np.empty((len(experiment.CHECKPOINTS), len(experiment.KS)))
    best_ks = []
    for i, checkpoint in enumerate(experiment.CHECKPOINTS):
        group = groups[checkpoint, count]
        xs = sorted(group)
        axes[0].plot(xs, [group[k]["fid"] for k in xs], marker="o", label=checkpoint)
        matrix[i] = [group[k]["fid"] - group[0]["fid"] for k in experiment.KS]
        best_ks.append(min(experiment.KS, key=lambda k: (group[k]["fid"], k)))
    axes[0].set(xlabel="K", ylabel="FID", title="FID by Restart K")
    axes[0].set_xticks(experiment.KS)
    axes[0].legend(fontsize=8, ncol=5, loc="upper center")
    draw_delta_heatmap(axes[1], matrix)

    if mean_k:
        draw_k_distribution(axes[2], groups, count)
    elif near_optimal:
        draw_near_optimal(axes[2], groups, count)
    else:
        axes[2].plot([int(checkpoint[:-1]) for checkpoint in experiment.CHECKPOINTS], best_ks, marker="o")
        axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
        axes[2].set_xticks([int(checkpoint[:-1]) for checkpoint in experiment.CHECKPOINTS],
                          experiment.CHECKPOINTS)
        axes[2].set_yticks(experiment.KS)
    draw_improvement(axes[3], groups, count, improvement_limits(groups))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def draw_delta_heatmap(axis, matrix):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FormatStrFormatter

    # An odd LUT length includes pure white exactly at normalized zero (0.5).
    colormap = LinearSegmentedColormap.from_list(
        "fid_delta", ["#3b4cc0", "#ffffff", "#b40426"], N=257)
    norm = Normalize(vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT)

    heatmap = axis.imshow(matrix, aspect="auto", cmap=colormap, norm=norm)
    axis.set_xticks(range(len(experiment.KS)), experiment.KS)
    axis.set_yticks(range(len(experiment.CHECKPOINTS)), experiment.CHECKPOINTS)
    axis.set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
    axis.figure.colorbar(heatmap, ax=axis, ticks=np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 9),
                        format=FormatStrFormatter("%.1f"))


def draw_near_optimal(axis, groups, count):
    """Show distance to the best evaluated K only for complete checkpoints."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    matrix = np.full((len(experiment.KS), len(experiment.CHECKPOINTS)), np.nan)
    best_x, best_y = [], []
    for i, checkpoint in enumerate(experiment.CHECKPOINTS):
        group = groups.get((checkpoint, count), {})
        if set(group) != set(experiment.KS):
            continue
        best = min(experiment.KS, key=lambda k: (group[k]["fid"], k))
        minimum = group[best]["fid"]
        matrix[:, i] = [group[k]["fid"] - minimum for k in experiment.KS]
        best_x.append(i)
        best_y.append(experiment.KS.index(best))

    colormap = LinearSegmentedColormap.from_list("near_optimal", ["#08306b", "#ffffff"])
    colormap.set_bad("#eeeeee")
    colormap.set_over("#ffffff")
    image = axis.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
                        cmap=colormap, norm=Normalize(vmin=0, vmax=0.05))
    axis.scatter(best_x, best_y, color="black", s=20, label="Best K", zorder=3)
    axis.set_xticks(range(len(experiment.CHECKPOINTS)), experiment.CHECKPOINTS)
    axis.set_yticks(range(len(experiment.KS)), experiment.KS)
    axis.set(xlabel="Training (kimg)", ylabel="K", title="K proximity to best FID")
    axis.legend(loc="upper right", fontsize=8)
    extend = "max" if np.any(matrix > 0.05) else "neither"
    colorbar = axis.figure.colorbar(image, ax=axis, ticks=[0, 0.01, 0.02, 0.05], extend=extend)
    colorbar.set_label("FID(K) - min FID")
    colorbar.ax.set_yticklabels(["0", "0.01", "0.02", "0.05"])


def fid_gap_within(delta, limit):
    """Include the decimal boundary despite floating-point subtraction error."""
    import math

    return delta <= limit or math.isclose(delta, limit, rel_tol=0, abs_tol=1e-12)


def near_optimal_candidates(group):
    minimum = min(row["fid"] for row in group.values())
    return [k for k in experiment.KS if fid_gap_within(group[k]["fid"] - minimum, 0.02)]


def fid_weighted_mean(ks, fids, temperature):
    """Average all K using normalized exp(-(FID - min FID) / temperature)."""
    import numpy as np

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("weight temperature must be finite and positive")
    values = np.asarray(fids, dtype=float)
    weights = np.exp(-(values - values.min()) / temperature)
    weights /= weights.sum()
    return float(np.dot(np.asarray(ks, dtype=float), weights))


def draw_k_distribution(axis, groups, count, weighted_temperature=None):
    """Share all points, mean line, labels and axes between both layouts."""
    import numpy as np

    close_x, close_y, near_x, near_y, best_x, best_y = [], [], [], [], [], []
    mean_ks = np.full(len(experiment.CHECKPOINTS), np.nan)
    for i, checkpoint in enumerate(experiment.CHECKPOINTS):
        group = groups.get((checkpoint, count), {})
        if set(group) != set(experiment.KS):
            axis.axvspan(i - 0.5, i + 0.5, color="#eeeeee", zorder=0)
            continue
        best = min(experiment.KS, key=lambda k: (group[k]["fid"], k))
        best_x.append(i)
        best_y.append(best)
        candidates = near_optimal_candidates(group)
        for k in experiment.KS:
            delta = group[k]["fid"] - group[best]["fid"]
            if fid_gap_within(delta, 0.01):
                close_x.append(i)
                close_y.append(k)
            elif fid_gap_within(delta, 0.02):
                near_x.append(i)
                near_y.append(k)
        if weighted_temperature is None:
            mean_ks[i] = np.mean(candidates)
        else:
            mean_ks[i] = fid_weighted_mean(
                experiment.KS, [group[k]["fid"] for k in experiment.KS], weighted_temperature)
    if weighted_temperature is None:
        axis.scatter(close_x, close_y, s=85, color="#2166ac", label="0 <= FID gap <= 0.01", zorder=2)
        axis.scatter(near_x, near_y, s=85, color="#a6cee3", label="0.01 < FID gap <= 0.02", zorder=2)
        mean_label = "Mean K (FID gap <= 0.02)"
    else:
        mean_label = f"FID-weighted mean K (tau={weighted_temperature:g})"
    axis.scatter(best_x, best_y, s=120, color="black", label="Best K (smallest K if tied)", zorder=3)
    axis.plot(range(len(experiment.CHECKPOINTS)), mean_ks, color="#ff7f0e",
              marker="o", markersize=4, linewidth=2, label=mean_label, zorder=4)
    axis.set_xticks(range(len(experiment.CHECKPOINTS)), experiment.CHECKPOINTS)
    axis.set_yticks(experiment.KS)
    title = ("Near-optimal K and mean: FID gap = FID(K) - min FID"
             if weighted_temperature is None else "Best K and FID-weighted mean")
    axis.set(xlabel="Training (kimg)", ylabel="K", title=title,
             xlim=(-0.5, len(experiment.CHECKPOINTS) - 0.5), ylim=(-0.5, 8.5))
    axis.set_axisbelow(True)
    axis.grid(alpha=0.15)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2)


def make_dots_figure(groups, count, weighted_temperature=None):
    """Draw exactly two data panels: baseline deltas and near-optimal K dots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    figure.suptitle(f"Experiment 8-11: FID-{count:,}, main=18, t_min=0.14", fontsize=14)
    matrix = np.full((len(experiment.CHECKPOINTS), len(experiment.KS)), np.nan)
    for i, checkpoint in enumerate(experiment.CHECKPOINTS):
        group = groups.get((checkpoint, count), {})
        if 0 in group:
            for j, k in enumerate(experiment.KS):
                if k in group:
                    matrix[i, j] = group[k]["fid"] - group[0]["fid"]
    draw_delta_heatmap(axes[0], matrix)
    draw_k_distribution(axes[1], groups, count, weighted_temperature)
    note = ("FID gap thresholds are display tolerances, not statistical equivalence."
            if weighted_temperature is None else
            f"All K included: weight proportional to exp(-(FID(K) - min FID) / {weighted_temperature:g}). "
            "Mean K describes the weighted center, not the optimal K.")
    figure.text(0.5, 0.02, note,
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure


def improvement_limits(groups):
    improvements = [group[0]["fid"] - row["fid"]
                    for group in groups.values() for k, row in group.items() if k]
    lower = min(0.0, min(improvements))
    upper = max(0.0, max(improvements))
    padding = 0.05 * (upper - lower or 1.0)
    return lower - padding, upper + padding


def draw_improvement(axis, groups, count, limits):
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    baselines = [groups[checkpoint, count][0]["fid"] for checkpoint in experiment.CHECKPOINTS]
    for k in experiment.KS:
        if k == 0:
            continue
        improvements = [baseline - groups[checkpoint, count][k]["fid"]
                        for checkpoint, baseline in zip(experiment.CHECKPOINTS, baselines)]
        axis.scatter(baselines, improvements, label=f"K={k}", color=colors[k - 1],
                     s=45, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    axis.axhline(0, color="0.3", linestyle="--", linewidth=1, zorder=2)
    axis.set(xlabel="FID(K=0)", ylabel="Improvement: FID(K=0) - FID(K)",
             title="Baseline FID vs improvement", ylim=limits)
    axis.margins(x=0.05)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, title="Restart K")


def make_improvement_figure(groups, count, limits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    draw_improvement(axis, groups, count, limits)
    axis.set_title(f"FID-{count:,}, main=18, t_min=0.14")
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_RESULTS / "restart_fid_summary.json",
                        help="merged summary JSON containing all 270 results")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS,
                        help="destination for PNG/SVG figures; result tables are never rewritten")
    parser.add_argument("--plot-kind", choices=("overview", "baseline-improvement", "near-optimal", "near-optimal-dots", "fid-weighted-k", "mean-k-overview", "all"), default="overview",
                        help="overview (default), baseline-improvement scatter, near-optimal overview, mean-k-overview (4 panels), near-optimal-dots or fid-weighted-k (2 panels), or both original plots (all)")
    parser.add_argument("--weight-temperature", type=float, default=0.02,
                        help="positive temperature in FID units for fid-weighted-k (default: 0.02)")
    args = parser.parse_args()
    if args.plot_kind == "fid-weighted-k":
        fid_weighted_mean(experiment.KS, [0.0] * len(experiment.KS), args.weight_temperature)
    groups = load_groups(args.summary)
    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RESULTS.parent / "logs" / "matplotlib"))
    limits = improvement_limits(groups)
    for count in experiment.STAGES:
        kinds = ("overview", "baseline-improvement") if args.plot_kind == "all" else (args.plot_kind,)
        for kind in kinds:
            if kind in ("overview", "near-optimal", "mean-k-overview"):
                figure = make_figure(groups, count, near_optimal=kind == "near-optimal",
                                     mean_k=kind == "mean-k-overview")
                stem = f"fixed_main18_tmin014_fid{count}"
                if kind == "near-optimal":
                    stem += "_near_optimal"
                elif kind == "mean-k-overview":
                    stem += "_mean_k"
            elif kind == "fid-weighted-k":
                figure = make_dots_figure(groups, count, weighted_temperature=args.weight_temperature)
                temperature_tag = str(args.weight_temperature).replace(".", "p")
                stem = f"fixed_main18_tmin014_fid{count}_fid_weighted_k_tau{temperature_tag}"
            elif kind == "near-optimal-dots":
                figure = make_dots_figure(groups, count)
                stem = f"fixed_main18_tmin014_fid{count}_near_optimal_dots"
            else:
                figure = make_improvement_figure(groups, count, limits)
                stem = f"fid_baseline_vs_improvement_fid{count}"
            for extension in ("png", "svg"):
                path = args.outdir / f"{stem}.{extension}"
                figure.savefig(path, dpi=180)
                print(path)
            import matplotlib.pyplot as plt
            plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
