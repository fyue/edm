#!/usr/bin/env python3
"""Plot Stage 0 fine S_churn sweep results."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/kyoukinkin/data/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SUMMARY = Path("experiments/4/stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_summary.tsv")
DEFAULT_OUT = Path("experiments/4/stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_curve.png")
ORDER = ["10k", "40k", "120k", "pretrained"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(summary: Path) -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with summary.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows[row["checkpoint"]].append((float(row["S_churn"]), float(row["fid_1k"])))
            except (KeyError, TypeError, ValueError):
                continue
    return {key: sorted(value) for key, value in rows.items() if value}


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary)
    if not rows:
        raise SystemExit(f"No plottable rows found in {args.summary}")

    fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    for key in ORDER + sorted(set(rows) - set(ORDER)):
        if key not in rows:
            continue
        xs = [x for x, _ in rows[key]]
        ys = [y for _, y in rows[key]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=key)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)

    ax.axvline(18 * (2**0.5 - 1), color="gray", linestyle="--", linewidth=1, label="saturation at steps=18")
    ax.set_title("Stage 0 third experiment: fine S_churn sweep below saturation")
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID-1k")
    ax.grid(True, alpha=0.3)
    ax.legend(title="checkpoint")
    ax.text(
        0.0,
        -0.18,
        "samples=1024, seeds=0-1023, steps=18; FID-1k is for relative screening only.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
