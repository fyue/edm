#!/usr/bin/env python3
"""Plot one progressive Experiment 8-6 FID stage."""

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
DEFAULT_OUTPUT = EXP_ROOT / "results" / "restart_k5_fid_overview"
CHECKPOINT_ORDER = [f"{kimg}k" for kimg in range(10, 121, 10)]
COLORS = {
    "main": "#4D4D4D",
    "equal_nfe": "#0072B2",
    "restart": "#D55E00",
    "improvement_main": "#E69F00",
    "improvement_nfe": "#009E73",
}
NUMERIC_FIELDS = [
    "restart_fid",
    "same_main_ode_fid",
    "same_nfe_ode_fid",
    "improvement_vs_same_main",
    "improvement_vs_same_nfe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--seeds", default="0-4999")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} in row: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field!r} in row: {row}")
    return value


def complete(row: dict[str, str]) -> bool:
    try:
        return all(math.isfinite(float(row[field])) for field in NUMERIC_FIELDS)
    except (KeyError, TypeError, ValueError):
        return False


def load_rows(path: Path, num_images: int, seeds: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing improvements: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        candidates = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("num_images") == str(num_images)
            and row.get("seeds") == seeds
            and row.get("restart_k") == "5"
            and complete(row)
        ]
    indexed = {row["checkpoint"]: row for row in candidates}
    rows = [indexed[checkpoint] for checkpoint in CHECKPOINT_ORDER if checkpoint in indexed]
    if not rows:
        raise SystemExit("No checkpoint has a complete Restart/baseline triplet")
    return rows


def label_bars(ax: plt.Axes, bars) -> None:
    ax.bar_label(
        bars,
        labels=[f"{bar.get_height():+.3f}" for bar in bars],
        padding=3,
        fontsize=7,
        rotation=90,
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.num_images, args.seeds)
    checkpoints = [row["checkpoint"] for row in rows]
    x = np.arange(len(rows))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4))

    absolute_series = [
        ("same_main_ode_fid", "ODE, 18 steps (35 NFE)", COLORS["main"], "o"),
        ("same_nfe_ode_fid", "ODE, 28 steps (55 NFE)", COLORS["equal_nfe"], "o"),
        ("restart_fid", "Restart K=5 (55 NFE)", COLORS["restart"], "s"),
    ]
    for field, label, color, marker in absolute_series:
        values = [number(row, field) for row in rows]
        axes[0].plot(
            x,
            values,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=label,
        )
    axes[0].set_title("(a) FID by training checkpoint")
    axes[0].set_ylabel(f"FID-{args.num_images:,} (lower is better)")
    axes[0].legend(frameon=True, fontsize=9)

    width = 0.36
    main_values = [number(row, "improvement_vs_same_main") for row in rows]
    nfe_values = [number(row, "improvement_vs_same_nfe") for row in rows]
    bars_main = axes[1].bar(
        x - width / 2,
        main_values,
        width,
        color=COLORS["improvement_main"],
        label="vs. 18-step ODE",
    )
    bars_nfe = axes[1].bar(
        x + width / 2,
        nfe_values,
        width,
        color=COLORS["improvement_nfe"],
        label="vs. equal-NFE ODE",
    )
    label_bars(axes[1], bars_main)
    label_bars(axes[1], bars_nfe)
    axes[1].axhline(0, color="black", linewidth=1.0, alpha=0.75)
    axes[1].set_title("(b) Paired FID reduction from Restart")
    axes[1].set_ylabel(
        f"FID-{args.num_images:,} reduction (positive favors Restart)"
    )
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].margins(y=0.24)

    for ax in axes:
        ax.set_xticks(x, checkpoints, rotation=45, ha="right")
        ax.set_xlabel("Training checkpoint")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Experiment 8-6: Restart K=5 FID-{args.num_images:,}",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        f"CIFAR-10 conditional; nested seed range {args.seeds}.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93), w_pad=2.5)

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
