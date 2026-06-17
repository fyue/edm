#!/usr/bin/env python3
"""Plot selected S_churn FID-5k confirmation results."""

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


DEFAULT_SUMMARY = Path("experiments/4/stage1/1_fid5k_selected_churn/results/fid5k_selected_churn_summary.tsv")
DEFAULT_OUT = Path("experiments/4/stage1/1_fid5k_selected_churn/results/fid5k_selected_churn_curve.png")
ORDER = ["5.5k", "10k", "15.0k", "20k", "40k", "80k", "120k", "pretrained"]


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
                rows[row["checkpoint"]].append((float(row["S_churn"]), float(row["fid_5k"])))
            except (KeyError, TypeError, ValueError):
                continue
    return {key: sorted(value) for key, value in rows.items() if value}


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary)
    if not rows:
        raise SystemExit(f"No plottable rows found in {args.summary}")

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for key in ORDER + sorted(set(rows) - set(ORDER)):
        if key not in rows:
            continue
        xs = [x for x, _ in rows[key]]
        ys = [y for _, y in rows[key]]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=key)

    ax.set_title("Stage 1 first experiment: selected S_churn FID-5k")
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID-5k")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="checkpoint",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )
    ax.text(
        0.0,
        -0.18,
        "samples=5000, seeds=0-4999, steps=18; FID-5k is still a screening metric.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
    )
    fig.subplots_adjust(right=0.78, bottom=0.16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
