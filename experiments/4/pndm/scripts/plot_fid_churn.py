#!/usr/bin/env python3
"""Plot FID against S_churn from one or more Experiment 4 PNDM summaries."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "edm-matplotlib-cache")
)

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="PNDM S_churn sensitivity")
    parser.add_argument("--seeds", help="Only plot this seed range")
    return parser.parse_args()


def load_rows(paths, seeds=None):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["sampler"] != "pndm":
                    continue
                if seeds is not None and row["seeds"] != seeds:
                    continue
                rows.append(row)
    return rows


def main():
    args = parse_args()
    rows = load_rows(args.summary, args.seeds)
    if not rows:
        raise SystemExit("No matching PNDM rows found")

    grouped = {}
    for row in rows:
        key = (row["checkpoint"], row["label"])
        grouped.setdefault(key, {})[float(row["S_churn"])] = float(row["fid"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (_, label), values in grouped.items():
        points = sorted(values.items())
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
