"""Redraw only the NFE-55 FID-50k figure, with a zero-centered heatmap."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "runs/nfe55/results"
CHECKPOINTS = ("20k", "40k", "60k", "80k", "100k")
KS = tuple(range(9))


def build_figure(results):
    groups = {}
    for row in results:
        if int(row["num_images"]) != 50000:
            continue
        group = groups.setdefault(row["checkpoint"], {})
        k = int(row["restart_k"])
        if k in group:
            raise ValueError("Duplicate FID-50k result")
        group[k] = float(row["fid"])
    if any(set(groups.get(checkpoint, {})) != set(KS) for checkpoint in CHECKPOINTS[:3]):
        raise ValueError("Expected all K=0..8 results for 20k, 40k and 60k")

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrix = np.full((len(CHECKPOINTS), len(KS)), np.nan)
    for i, checkpoint in enumerate(CHECKPOINTS):
        group = groups.get(checkpoint, {})
        if not group:
            continue
        xs = sorted(group)
        axes[0].plot(xs, [group[k] for k in xs], marker="o", label=checkpoint)
        if 0 in group:
            for j, k in enumerate(KS):
                if k in group:
                    matrix[i, j] = group[k] - group[0]
    axes[0].set(xlabel="K", ylabel="FID", title="FID-50000, nfe55, subset30k, t_min=0.14")
    axes[0].set_xticks(KS)
    axes[0].legend(fontsize=8)
    heatmap = axes[1].imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="coolwarm")
    axes[1].set_xticks(range(len(KS)), KS)
    axes[1].set_yticks(range(len(CHECKPOINTS)), CHECKPOINTS)
    axes[1].set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
    colorbar = figure.colorbar(heatmap, ax=axes[1])
    complete = [checkpoint for checkpoint in CHECKPOINTS if set(groups.get(checkpoint, {})) == set(KS)]
    axes[2].plot([int(checkpoint[:-1]) for checkpoint in complete],
                 [min(KS, key=lambda k: (groups[checkpoint][k], k)) for checkpoint in complete], marker="o")
    axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
    axes[2].set_yticks(KS)

    # Resolve the original layout first so the side panels stay in place.
    figure.tight_layout()
    side_positions = [axes[index].get_position().bounds for index in (0, 2)]
    selected = matrix[:3]
    limit = np.ceil(np.nanmax(np.abs(selected)) * 10) / 10
    limit = max(float(limit), 0.1)
    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#2166AC", "#FFFFFF", "#B2182B"], N=257)
    heatmap.set_data(np.ma.masked_invalid(selected))
    heatmap.set_extent((-0.5, len(KS) - 0.5, 2.5, -0.5))
    heatmap.set_cmap(cmap)
    heatmap.set_norm(TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[1].set_yticks(range(3), CHECKPOINTS[:3])
    colorbar.update_normal(heatmap)
    colorbar.set_ticks(np.linspace(-limit, limit, 2 * round(limit * 10) + 1))
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    assert tuple(cmap(heatmap.norm(0))) == (1.0, 1.0, 1.0, 1.0)
    assert side_positions == [axes[index].get_position().bounds for index in (0, 2)]
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = json.loads((args.results_dir / "restart_fid_summary.json").read_text(encoding="utf-8"))
    figure = build_figure(results)
    for extension in ("png", "svg"):
        output = args.results_dir / f"fixed_nfe55_tmin014_fid50000.{extension}"
        figure.savefig(output, dpi=180)
        print(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
