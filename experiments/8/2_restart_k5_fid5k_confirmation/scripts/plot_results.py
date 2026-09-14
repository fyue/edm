#!/usr/bin/env python3
"""Plot Experiment 8-2 Restart FID improvements for all evaluated K values."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/kyoukinkin/data/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXP_ROOT / "results" / "restart_improvements.tsv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "restart_fid5k_improvements"
CHECKPOINT_ORDER = ["5.5k", "20k", "120k", "200k"]
K_ORDER = [1, 2, 5]
K_COLORS = {1: "#0072B2", 2: "#E69F00", 5: "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--seeds", default="0-4999")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def parse_number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} in row: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field!r} in row: {row}")
    return value


def load_rows(
    path: Path, num_images: int, seeds: str
) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        candidates = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("num_images") == str(num_images) and row.get("seeds") == seeds
        ]

    rows: dict[tuple[str, int], dict[str, str]] = {}
    for row in candidates:
        try:
            key = row["checkpoint"], int(row["restart_k"])
            for field in ["improvement_vs_same_main", "improvement_vs_same_nfe"]:
                parse_number(row, field)
        except (KeyError, TypeError, ValueError):
            continue
        if key in rows:
            raise ValueError(f"Duplicate result for checkpoint/K: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(
            f"No complete rows for num_images={num_images}, seeds={seeds} in {path}"
        )
    return rows


def add_value_labels(ax: plt.Axes, x_values: list[float], values: list[float]) -> None:
    for x_value, value in zip(x_values, values):
        offset = 7 if value >= 0 else -10
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            f"{value:+.3f}",
            (x_value, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_panel(
    ax: plt.Axes,
    rows: dict[tuple[str, int], dict[str, str]],
    field: str,
    title: str,
) -> None:
    x = np.arange(len(CHECKPOINT_ORDER), dtype=float)
    offsets = {1: -0.14, 2: 0.0, 5: 0.14}
    for restart_k in K_ORDER:
        points = [
            (index, parse_number(rows[(checkpoint, restart_k)], field))
            for index, checkpoint in enumerate(CHECKPOINT_ORDER)
            if (checkpoint, restart_k) in rows
        ]
        if not points:
            continue
        xs = [float(index) + offsets[restart_k] for index, _ in points]
        values = [value for _, value in points]
        ax.plot(
            xs,
            values,
            color=K_COLORS[restart_k],
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=f"K={restart_k}",
        )
        add_value_labels(ax, xs, values)

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.75)
    ax.set_title(title)
    ax.set_xticks(x, CHECKPOINT_ORDER)
    ax.set_xlabel("Training checkpoint")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.18)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.num_images, args.seeds)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    plot_panel(
        axes[0],
        rows,
        "improvement_vs_same_main",
        "(a) Improvement over 18-step ODE",
    )
    plot_panel(
        axes[1],
        rows,
        "improvement_vs_same_nfe",
        "(b) Improvement over equal-NFE ODE",
    )
    axes[0].set_ylabel(
        f"FID-{args.num_images:,} reduction (positive favors Restart)"
    )
    axes[1].legend(title="Restart setting", frameon=True)

    fig.suptitle("Experiment 8-2: Restart benefit by K and checkpoint", fontsize=14)
    fig.text(
        0.5,
        0.015,
        f"CIFAR-10 conditional; {args.num_images:,} images, seeds {args.seeds}. "
        f"FID-{args.num_images:,} is not the final three-seed estimate.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93), w_pad=2.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    svg_path = args.output.with_suffix(".svg")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
