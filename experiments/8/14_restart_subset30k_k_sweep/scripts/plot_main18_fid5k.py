"""Redraw one main18 FID stage for 20k/40k/60k, with exact white at zero."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "runs/main18/results"
CHECKPOINTS = ("20k", "40k", "60k")
KS = tuple(range(9))
STAGES = (5000, 25000, 50000)


def figure_name(num_images):
    return f"fixed_main18_tmin014_fid{num_images}"


def selected_groups(results, num_images=5000):
    if num_images not in STAGES:
        raise ValueError("Expected FID-5k, FID-25k or FID-50k")
    groups = {checkpoint: {} for checkpoint in CHECKPOINTS}
    for row in results:
        checkpoint = row["checkpoint"]
        if checkpoint not in groups or row["num_images"] != num_images:
            continue
        k = row["restart_k"]
        if type(k) is not int or k not in KS:
            raise ValueError("Expected integer K=0..8")
        if row["condition"] != f"{checkpoint}-k{k}" or row["seeds"] != f"0-{num_images - 1}":
            raise ValueError("Result condition/seed range does not match FID stage")
        if row["main_steps"] != 18:
            raise ValueError("Expected main18 result")
        fid = float(row["fid"])
        if not math.isfinite(fid) or fid < 0:
            raise ValueError("FID must be finite and nonnegative")
        if k in groups[checkpoint]:
            raise ValueError(f"Duplicate FID-{num_images} result: {checkpoint}-k{k}")
        groups[checkpoint][k] = fid
    if any(set(group) != set(KS) for group in groups.values()):
        raise ValueError("Expected 27 results: 20k/40k/60k x K=0..8")
    return groups


def build_figure(results, num_images=5000):
    groups = selected_groups(results, num_images)
    matrix = np.array([[groups[checkpoint][k] - groups[checkpoint][0] for k in KS]
                       for checkpoint in CHECKPOINTS])
    limit = float(np.max(np.abs(matrix))) or 1.0
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#2166AC", "#FFFFFF", "#B2182B"], N=257)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for checkpoint in CHECKPOINTS:
        axes[0].plot(KS, [groups[checkpoint][k] for k in KS], marker="o", label=checkpoint)
    axes[0].set(xlabel="K", ylabel="FID", title=f"FID-{num_images}, main18, subset30k, t_min=0.14")
    axes[0].set_xticks(KS)
    axes[0].legend(fontsize=8)

    heatmap = axes[1].imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    axes[1].set_xticks(range(len(KS)), KS)
    axes[1].set_yticks(range(len(CHECKPOINTS)), CHECKPOINTS)
    axes[1].set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
    colorbar = figure.colorbar(heatmap, ax=axes[1])
    colorbar.set_ticks(np.linspace(-limit, limit, 5))
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: "0.0" if value == 0 else f"{value:.3f}"))

    axes[2].plot([int(checkpoint[:-1]) for checkpoint in CHECKPOINTS],
                 [min(KS, key=lambda k: (groups[checkpoint][k], k)) for checkpoint in CHECKPOINTS], marker="o")
    axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
    axes[2].set_xticks([int(checkpoint[:-1]) for checkpoint in CHECKPOINTS])
    axes[2].set_yticks(KS)
    figure.tight_layout()
    return figure


def redraw(results_dir, num_images=5000):
    results = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(results_dir.glob(f"*-fid{num_images}.json"))]
    figure = build_figure(results, num_images)
    try:
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            for extension in ("png", "svg"):
                output = results_dir / f"{figure_name(num_images)}.{extension}"
                figure.savefig(output, dpi=180)
                print(output)
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--num-images", type=int, choices=STAGES, default=5000)
    args = parser.parse_args()
    redraw(args.results_dir, args.num_images)


if __name__ == "__main__":
    main()
