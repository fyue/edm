#!/usr/bin/env python3
"""Merge Stage 2 experiment 1 with S_churn=6,7 and remake result plots."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXP_ROOT = Path("experiments/4/stage2/3_fid50k_churn_6_7")
DEFAULT_BASE = Path(
    "experiments/4/stage2/1_fid50k_selected_churn/results/"
    "fid50k_selected_churn_summary.tsv"
)
DEFAULT_NEW = EXP_ROOT / "results/fid50k_selected_churn_summary.tsv"
DEFAULT_COMBINED = EXP_ROOT / "results/fid50k_churn_0_7_combined.tsv"
DEFAULT_LOW = EXP_ROOT / "results/fid50k_churn_0_7_low_fid.tsv"
DEFAULT_BEST = EXP_ROOT / "results/fid50k_churn_0_7_best.tsv"
DEFAULT_CURVE = EXP_ROOT / "results/fid50k_churn_0_7_curve.png"
DEFAULT_LOW_CURVE = EXP_ROOT / "results/fid50k_churn_0_7_curve_low_fid.png"

HEADER = [
    "checkpoint", "label", "S_churn", "num_images", "seeds", "steps",
    "S_min", "S_max", "S_noise", "fid_50k", "network", "images", "log",
]
ORDER = ["5.5k", "10k", "20k", "40k", "80k", "120k", "pretrained"]
LOW_FID_KEYS = ["40k", "80k", "120k", "pretrained"]
COLORS = {
    "5.5k": "#1f77b4", "10k": "#ff7f0e", "20k": "#2ca02c",
    "40k": "#d62728", "80k": "#7b2cbf", "120k": "#8c564b",
    "pretrained": "#4d4d4d",
}
MARKERS = {
    "5.5k": "o", "10k": "s", "20k": "^", "40k": "v",
    "80k": "D", "120k": "P", "pretrained": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--new-summary", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--low-fid-out", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--best-out", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--curve-out", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--low-curve-out", type=Path, default=DEFAULT_LOW_CURVE)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing summary: {path}")
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != HEADER:
            raise SystemExit(f"Unexpected header in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                float(row["S_churn"])
                float(row["fid_50k"])
            except (TypeError, ValueError):
                continue
            rows.append(dict(row))
    return rows


def condition_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] for key in (
        "checkpoint", "S_churn", "num_images", "seeds", "steps",
        "S_min", "S_max", "S_noise",
    ))


def sort_key(row: dict[str, str]) -> tuple[int, float]:
    checkpoint = row["checkpoint"]
    index = ORDER.index(checkpoint) if checkpoint in ORDER else len(ORDER)
    return index, float(row["S_churn"])


def merge_rows(base: list[dict[str, str]], new: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = {condition_key(row): row for row in base}
    for row in new:
        key = condition_key(row)
        if key in merged and merged[key]["fid_50k"] != row["fid_50k"]:
            raise SystemExit(f"Conflicting duplicate condition: {key}")
        merged[key] = row
    return sorted(merged.values(), key=sort_key)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def group_points(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint"]].append((float(row["S_churn"]), float(row["fid_50k"])))
    return {key: sorted(points) for key, points in grouped.items()}


def write_best(path: Path, grouped: dict[str, list[tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["checkpoint", "best_S_churn", "best_fid_50k"])
        for checkpoint in ORDER:
            if checkpoint in grouped:
                churn, fid = min(grouped[checkpoint], key=lambda point: point[1])
                writer.writerow([checkpoint, f"{churn:g}", f"{fid:.8g}"])
    print(f"Wrote {path}")


def plot_curve(
    grouped: dict[str, list[tuple[float, float]]],
    checkpoints: list[str],
    out: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    values: list[float] = []
    for checkpoint in checkpoints:
        points = grouped.get(checkpoint)
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        values.extend(ys)
        ax.plot(xs, ys, marker=MARKERS.get(checkpoint, "o"),
                color=COLORS.get(checkpoint), linewidth=2, markersize=6,
                label=checkpoint)
    if not values:
        raise SystemExit("No plottable rows found")
    ax.set_title(title)
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID-50k")
    ax.set_xlim(-0.2, 7.2)
    ax.set_xticks(range(8))
    ax.grid(True, alpha=0.3)
    ax.axvspan(5.5, 7.2, color="#eeeeee", alpha=0.7, zorder=0)
    ax.text(6.25, 0.98, "new: S=6,7", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="dimgray")
    if checkpoints == LOW_FID_KEYS:
        padding = max((max(values) - min(values)) * 0.10, 0.03)
        ax.set_ylim(min(values) - padding, max(values) + padding)
    ax.legend(title="checkpoint", loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0, frameon=True)
    ax.text(0.0, -0.18, "samples=50000, seeds=0-49999, steps=18.",
            transform=ax.transAxes, fontsize=8, color="dimgray")
    fig.subplots_adjust(right=0.78, bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    args = parse_args()
    combined = merge_rows(read_rows(args.base_summary), read_rows(args.new_summary))
    low_rows = [row for row in combined if row["checkpoint"] in LOW_FID_KEYS]
    write_rows(args.combined_out, combined)
    write_rows(args.low_fid_out, low_rows)
    grouped = group_points(combined)
    write_best(args.best_out, grouped)
    plot_curve(grouped, ORDER, args.curve_out,
               "Stage 2: FID-50k vs S_churn, combined selected S and S=6,7", args.dpi)
    plot_curve(grouped, LOW_FID_KEYS, args.low_curve_out,
               "Stage 2: FID-50k vs S_churn, low-FID checkpoints", args.dpi)


if __name__ == "__main__":
    main()
