#!/usr/bin/env python3
"""Summarize Stage 2 FID-50k representative pairs over three seed ranges."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/kyoukinkin/data/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_SUMMARY = Path("experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_summary.tsv")
EXTRA_SUMMARY = Path("experiments/4/stage2/2_fid50k_extra_seeds/results/fid50k_extra_seeds_summary.tsv")
OUT_DIR = Path("experiments/4/stage2/2_fid50k_extra_seeds/results")
ORDER = ["5.5k", "10k", "20k", "40k", "80k", "120k", "pretrained"]
BEST_CHURN = {
    "5.5k": 5.0,
    "10k": 3.0,
    "20k": 3.0,
    "40k": 2.0,
    "80k": 3.0,
    "120k": 3.0,
    "pretrained": 3.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, default=BASE_SUMMARY)
    parser.add_argument("--extra-summary", type=Path, default=EXTRA_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows.extend(row for row in reader if row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def fmt(value: float) -> str:
    return f"{value:.6g}"


def representative_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        checkpoint = row.get("checkpoint", "")
        try:
            churn = float(row.get("S_churn", "nan"))
            num_images = int(row.get("num_images", "0"))
            steps = int(row.get("steps", "0"))
        except ValueError:
            continue
        if checkpoint not in BEST_CHURN:
            continue
        if num_images != 50000 or steps != 18:
            continue
        if churn not in (0.0, BEST_CHURN[checkpoint]):
            continue
        selected.append(row)
    return selected


def write_condition_stats(rows: list[dict[str, str]], out: Path) -> None:
    grouped: dict[tuple[str, float], list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        key = (row["checkpoint"], float(row["S_churn"]))
        grouped[key].append((row["seeds"], float(row["fid_50k"])))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["checkpoint", "S_churn", "n", "fid_mean", "fid_std", "values"])
        for checkpoint in ORDER:
            for churn in [0.0, BEST_CHURN[checkpoint]]:
                values = [fid for _, fid in sorted(grouped.get((checkpoint, churn), []))]
                if not values:
                    continue
                writer.writerow([
                    checkpoint,
                    f"{churn:g}",
                    len(values),
                    fmt(mean(values)),
                    fmt(sample_std(values)),
                    ",".join(fmt(value) for value in values),
                ])


def paired_improvements(rows: list[dict[str, str]]) -> dict[str, list[tuple[str, float]]]:
    by_key: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for row in rows:
        checkpoint = row["checkpoint"]
        seeds = row["seeds"]
        churn = float(row["S_churn"])
        by_key[(checkpoint, seeds)][churn] = float(row["fid_50k"])

    result: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for checkpoint in ORDER:
        best = BEST_CHURN[checkpoint]
        for (ckpt, seeds), values in sorted(by_key.items()):
            if ckpt != checkpoint:
                continue
            if 0.0 in values and best in values:
                result[checkpoint].append((seeds, values[0.0] - values[best]))
    return result


def write_improvement_stats(rows: list[dict[str, str]], out: Path) -> None:
    improvements = paired_improvements(rows)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["checkpoint", "best_S_churn", "n", "improvement_mean", "improvement_std", "values"])
        for checkpoint in ORDER:
            values = [value for _, value in improvements.get(checkpoint, [])]
            if not values:
                continue
            writer.writerow([
                checkpoint,
                f"{BEST_CHURN[checkpoint]:g}",
                len(values),
                fmt(mean(values)),
                fmt(sample_std(values)),
                ",".join(fmt(value) for value in values),
            ])


def plot_improvements(rows: list[dict[str, str]], out: Path, dpi: int) -> None:
    improvements = paired_improvements(rows)
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for checkpoint in ORDER:
        values = [value for _, value in improvements.get(checkpoint, [])]
        if not values:
            continue
        labels.append(checkpoint)
        means.append(mean(values))
        stds.append(sample_std(values))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4c78a8")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Stage 2 experiment 2: paired FID-50k improvement")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("FID-50k improvement, S=0 - best S")
    ax.grid(True, axis="y", alpha=0.3)
    for idx, (mu, sd) in enumerate(zip(means, stds)):
        ax.annotate(
            f"{mu:.3g}\n±{sd:.2g}",
            (idx, mu),
            textcoords="offset points",
            xytext=(0, 4 if mu >= 0 else -4),
            ha="center",
            va="bottom" if mu >= 0 else "top",
            fontsize=8,
        )
    fig.subplots_adjust(bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Wrote {out}")


def main() -> None:
    args = parse_args()
    rows = representative_rows(read_rows([args.base_summary, args.extra_summary]))
    if not rows:
        raise SystemExit("No representative rows found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    condition_stats = args.out_dir / "fid50k_3seed_condition_stats.tsv"
    improvement_stats = args.out_dir / "fid50k_3seed_improvement_stats.tsv"
    improvement_plot = args.out_dir / "fid50k_3seed_improvement.png"

    write_condition_stats(rows, condition_stats)
    write_improvement_stats(rows, improvement_stats)
    plot_improvements(rows, improvement_plot, args.dpi)
    print(f"Wrote {condition_stats}")
    print(f"Wrote {improvement_stats}")


if __name__ == "__main__":
    main()
