#!/usr/bin/env python3
"""Summarize Experiment 5-3 RK4 FID-50k over seed ranges."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_SUMMARY = Path("experiments/5/3_rk4_fid50k_3seeds/results/rk4_fid50k_3seeds_summary.tsv")
DEFAULT_OUT = Path("experiments/5/3_rk4_fid50k_3seeds/results/rk4_fid50k_3seeds_stats.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def fmt(value: float) -> str:
    return f"{value:.6g}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing summary: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f, delimiter="\t") if row]


def write_stats(rows: list[dict[str, str]], out: Path) -> None:
    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        key = (
            row["checkpoint"],
            row["label"],
            row["sampler"],
            row["S_churn"],
            row["num_images"],
            row["steps"],
        )
        grouped[key].append((row["seeds"], float(row["fid_50k"])))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "checkpoint",
            "label",
            "sampler",
            "S_churn",
            "num_images",
            "steps",
            "n",
            "fid_mean",
            "fid_std",
            "fid_min",
            "seed_fids",
        ])
        for key in sorted(grouped):
            seed_values = sorted(grouped[key])
            values = [fid for _, fid in seed_values]
            writer.writerow([
                *key,
                len(values),
                fmt(mean(values)),
                fmt(sample_std(values)),
                fmt(min(values)),
                ",".join(f"{seed}:{fmt(fid)}" for seed, fid in seed_values),
            ])


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary)
    if not rows:
        raise SystemExit("No rows found.")
    write_stats(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
