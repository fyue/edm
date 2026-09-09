#!/usr/bin/env python3
"""Merge Stage 1 FID-5k results and plot the extended S_churn sweep."""

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


EXP_ROOT = Path("experiments/4/stage1/2_fid5k_churn_6_7")
DEFAULT_BASE_SUMMARY = Path(
    "experiments/4/stage1/1_fid5k_selected_churn/results/"
    "fid5k_selected_churn_summary.tsv"
)
DEFAULT_NEW_SUMMARY = EXP_ROOT / "results/fid5k_selected_churn_summary.tsv"
DEFAULT_COMBINED = EXP_ROOT / "results/fid5k_churn_0_7_combined.tsv"
DEFAULT_LOW_FID = EXP_ROOT / "results/fid5k_churn_0_7_low_fid.tsv"
DEFAULT_BEST = EXP_ROOT / "results/fid5k_churn_0_7_best.tsv"
DEFAULT_CURVE = EXP_ROOT / "results/fid5k_churn_0_7_curve.png"
DEFAULT_LOW_CURVE = EXP_ROOT / "results/fid5k_churn_0_7_curve_low_fid.png"

HEADER = [
    "checkpoint",
    "label",
    "S_churn",
    "num_images",
    "seeds",
    "steps",
    "S_min",
    "S_max",
    "S_noise",
    "fid_5k",
    "network",
    "images",
    "log",
]
ORDER = ["5.5k", "10k", "15.0k", "20k", "40k", "80k", "120k", "pretrained"]
LOW_FID_KEYS = ["40k", "80k", "120k", "pretrained"]
COLORS = {
    "5.5k": "#1f77b4",
    "10k": "#ff7f0e",
    "15.0k": "#2ca02c",
    "20k": "#d62728",
    "40k": "#9467bd",
    "80k": "#8c564b",
    "120k": "#e377c2",
    "pretrained": "#4d4d4d",
}
MARKERS = {
    "5.5k": "o",
    "10k": "s",
    "15.0k": "^",
    "20k": "v",
    "40k": "D",
    "80k": "P",
    "120k": "X",
    "pretrained": "*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE_SUMMARY)
    parser.add_argument("--new-summary", type=Path, default=DEFAULT_NEW_SUMMARY)
    parser.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--low-fid-out", type=Path, default=DEFAULT_LOW_FID)
    parser.add_argument("--best-out", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--curve-out", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--low-curve-out", type=Path, default=DEFAULT_LOW_CURVE)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing summary: {path}")

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for values in reader:
            if len(values) < len(HEADER):
                continue
            row = dict(zip(HEADER, values[: len(HEADER)]))
            try:
                float(row["S_churn"])
                float(row["fid_5k"])
            except ValueError:
                continue
            rows.append(row)
    return rows


def condition_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] for key in [
        "checkpoint", "S_churn", "num_images", "seeds", "steps",
        "S_min", "S_max", "S_noise",
    ])


def row_sort_key(row: dict[str, str]) -> tuple[int, float]:
    checkpoint = row["checkpoint"]
    checkpoint_index = ORDER.index(checkpoint) if checkpoint in ORDER else len(ORDER)
    return checkpoint_index, float(row["S_churn"])


def merge_rows(base_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = {condition_key(row): row for row in base_rows}
    for row in new_rows:
        key = condition_key(row)
        if key in merged and merged[key]["fid_5k"] != row["fid_5k"]:
            raise SystemExit(f"Conflicting duplicate condition: {key}")
        merged[key] = row
    return sorted(merged.values(), key=row_sort_key)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def group_points(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint"]].append((float(row["S_churn"]), float(row["fid_5k"])))
    return {key: sorted(points) for key, points in grouped.items()}


def write_best(path: Path, grouped: dict[str, list[tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["checkpoint", "best_S_churn", "best_fid_5k"])
        for checkpoint in ORDER:
            if checkpoint not in grouped:
                continue
            churn, fid = min(grouped[checkpoint], key=lambda point: point[1])
            writer.writerow([checkpoint, f"{churn:g}", f"{fid:.8g}"])
    print(f"Wrote {path}")


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID-5k")
    ax.set_xlim(-0.2, 7.2)
    ax.set_xticks(range(8))
    ax.grid(True, alpha=0.3)
    ax.axvspan(5.5, 7.2, color="#eeeeee", alpha=0.7, zorder=0)
    ax.text(
        6.25,
        0.98,
        "new: S=6,7",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color="dimgray",
    )


def plot_curve(
    grouped: dict[str, list[tuple[float, float]]],
    checkpoints: list[str],
    out: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    plotted_values: list[float] = []
    for checkpoint in checkpoints:
        points = grouped.get(checkpoint)
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plotted_values.extend(ys)
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(checkpoint, "o"),
            color=COLORS.get(checkpoint),
            linewidth=2,
            markersize=6,
            label=checkpoint,
        )

    if not plotted_values:
        raise SystemExit("No plottable rows found")

    style_axis(ax, title)
    if checkpoints == LOW_FID_KEYS:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        padding = max((y_max - y_min) * 0.10, 0.03)
        ax.set_ylim(y_min - padding, y_max + padding)
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
        "samples=5000, seeds=0-4999, steps=18; FID-5k is a screening metric.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
    )
    fig.subplots_adjust(right=0.78, bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    args = parse_args()
    base_rows = read_rows(args.base_summary)
    new_rows = read_rows(args.new_summary)
    combined_rows = merge_rows(base_rows, new_rows)
    low_fid_rows = [row for row in combined_rows if row["checkpoint"] in LOW_FID_KEYS]

    write_rows(args.combined_out, combined_rows)
    write_rows(args.low_fid_out, low_fid_rows)

    grouped = group_points(combined_rows)
    write_best(args.best_out, grouped)
    plot_curve(
        grouped,
        ORDER,
        args.curve_out,
        "Stage 1: FID-5k vs S_churn, combined S=0-7",
        args.dpi,
    )
    plot_curve(
        grouped,
        LOW_FID_KEYS,
        args.low_curve_out,
        "Stage 1: FID-5k vs S_churn, low-FID checkpoints",
        args.dpi,
    )


if __name__ == "__main__":
    main()
