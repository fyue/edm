#!/usr/bin/env python3
"""Plot Experiment 8-4 FID-by-K curves and the selected K trajectory."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/kyoukinkin/data/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = EXP_ROOT / "results" / "restart_k_sweep.tsv"
DEFAULT_OPTIMAL = EXP_ROOT / "results" / "optimal_k.tsv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "restart_k_sweep_overview"
COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F7F7F",
    "#000000",
    "#A65628",
    "#17BECF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--optimal", type=Path, default=DEFAULT_OPTIMAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    args = parse_args()
    sweep_rows = [row for row in read_tsv(args.sweep) if row["complete_sweep"] == "1"]
    optimal_rows = read_tsv(args.optimal)
    if not optimal_rows:
        raise ValueError("No complete K=0..8 sweep is available to plot")
    protocols = {
        (row["num_images"], row["seeds"])
        for row in [*sweep_rows, *optimal_rows]
    }
    if len(protocols) != 1:
        raise ValueError(f"Expected one FID protocol, got: {protocols}")
    protocol = optimal_rows[0]
    num_images = int(protocol["num_images"])

    checkpoints = sorted(
        {row["checkpoint"] for row in sweep_rows},
        key=lambda checkpoint: int(
            next(row["training_kimg"] for row in sweep_rows if row["checkpoint"] == checkpoint)
        ),
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))

    for index, checkpoint in enumerate(checkpoints):
        selected = sorted(
            (row for row in sweep_rows if row["checkpoint"] == checkpoint),
            key=lambda row: int(row["restart_k"]),
        )
        restart_k = [int(row["restart_k"]) for row in selected]
        fid = [float(row["fid"]) for row in selected]
        delta = [float(row["delta_vs_k0"]) for row in selected]
        color = COLORS[index % len(COLORS)]
        axes[0].plot(restart_k, fid, marker="o", linewidth=2, color=color, label=checkpoint)
        axes[1].plot(restart_k, delta, marker="o", linewidth=2, color=color, label=checkpoint)

    axes[0].set_title("(a) FID across Restart K")
    axes[0].set_xlabel("Restart K")
    axes[0].set_xticks(range(9))
    axes[0].grid(alpha=0.25)
    axes[0].legend(title="Checkpoint")

    axes[1].axhline(0, color="black", linewidth=1, alpha=0.7)
    axes[1].set_title("(b) Change from K=0 ODE")
    axes[1].set_xlabel("Restart K")
    axes[1].set_ylabel("FID(K) - FID(K=0); lower is better")
    axes[1].set_xticks(range(9))
    axes[1].grid(alpha=0.25)

    optimal_rows.sort(key=lambda row: int(row["training_kimg"]))
    training_kimg = [int(row["training_kimg"]) for row in optimal_rows]
    optimal_k = [int(row["optimal_k"]) for row in optimal_rows]
    axes[2].plot(training_kimg, optimal_k, marker="o", linewidth=2, color="#6A3D9A")
    for x_value, y_value in zip(training_kimg, optimal_k):
        axes[2].annotate(
            f"K={y_value}",
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axes[2].set_title("(c) FID-minimizing K")
    axes[2].set_xlabel("Training checkpoint (kimg)")
    axes[2].set_ylabel("Selected K")
    axes[2].set_xticks(training_kimg, [row["checkpoint"] for row in optimal_rows])
    axes[2].set_yticks(range(9))
    axes[2].set_ylim(-0.4, 8.4)
    axes[2].grid(alpha=0.25)

    axes[0].set_ylabel(f"FID-{num_images:,} (lower is better)")
    fig.suptitle(
        f"Experiment 8-4: FID-{num_images:,} Restart K sweep over training progress",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        f"CIFAR-10 conditional; {num_images:,} images, "
        f"seeds {protocol['seeds']}. Screening result; K is selected by minimum FID.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93), w_pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        path = args.output.with_suffix(suffix)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
