#!/usr/bin/env python3
"""Plot Stage 2 selected S_churn FID-50k results."""

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


DEFAULT_SUMMARY = Path("experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_summary.tsv")
DEFAULT_CURVE_OUT = Path("experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_curve.png")
DEFAULT_ZOOM_OUT = Path("experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_curve_zoom_low.png")
DEFAULT_IMPROVEMENT_OUT = Path("experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_improvement.png")
ORDER = ["5.5k", "10k", "20k", "40k", "80k", "120k", "pretrained"]
COLORS = {
    "5.5k": "#1f77b4",
    "10k": "#ff7f0e",
    "20k": "#2ca02c",
    "40k": "#d62728",
    "80k": "#7b2cbf",
    "120k": "#8c564b",
    "pretrained": "#e377c2",
}
MARKERS = {
    "5.5k": "o",
    "10k": "s",
    "20k": "^",
    "40k": "v",
    "80k": "D",
    "120k": "P",
    "pretrained": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--curve-out", type=Path, default=DEFAULT_CURVE_OUT)
    parser.add_argument("--zoom-out", type=Path, default=DEFAULT_ZOOM_OUT)
    parser.add_argument("--improvement-out", type=Path, default=DEFAULT_IMPROVEMENT_OUT)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(summary: Path) -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with summary.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows[row["checkpoint"]].append((float(row["S_churn"]), float(row["fid_50k"])))
            except (KeyError, TypeError, ValueError):
                continue
    return {key: sorted(value) for key, value in rows.items() if value}


def ordered_keys(rows: dict[str, list[tuple[float, float]]]) -> list[str]:
    return [key for key in ORDER if key in rows] + sorted(set(rows) - set(ORDER))


def plot_curve(rows: dict[str, list[tuple[float, float]]], out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for key in ordered_keys(rows):
        xs = [x for x, _ in rows[key]]
        ys = [y for _, y in rows[key]]
        is_80k = key == "80k"
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(key, "o"),
            color=COLORS.get(key),
            linewidth=3 if is_80k else 2,
            markersize=7 if is_80k else 5,
            label=key,
            zorder=5 if is_80k else 3,
        )

    ax.set_title("Stage 2: selected S_churn FID-50k")
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID-50k")
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
        "samples=50000, seeds=0-49999, steps=18.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
    )
    fig.subplots_adjust(right=0.78, bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Wrote {out}")


def plot_low_zoom(rows: dict[str, list[tuple[float, float]]], out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    zoom_keys = [key for key in ["40k", "80k", "120k", "pretrained"] if key in rows]
    for key in zoom_keys:
        xs = [x for x, _ in rows[key]]
        ys = [y for _, y in rows[key]]
        is_80k = key == "80k"
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(key, "o"),
            color=COLORS.get(key),
            linewidth=3 if is_80k else 2,
            markersize=8 if is_80k else 6,
            label=key,
            zorder=5 if is_80k else 3,
        )
        ax.annotate(
            key,
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            ha="left",
            va="center",
            fontsize=9,
            color=COLORS.get(key),
        )

    ax.set_title("Stage 2: selected S_churn FID-50k, low-FID zoom")
    ax.set_xlabel("S_churn")
    ax.set_ylabel("FID-50k")
    ax.set_xlim(-0.15, 5.35)
    ax.set_ylim(1.78, 2.42)
    ax.grid(True, alpha=0.3)
    ax.legend(title="checkpoint", loc="upper right", frameon=True)
    fig.subplots_adjust(right=0.88, bottom=0.14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Wrote {out}")


def plot_improvement(rows: dict[str, list[tuple[float, float]]], out: Path, dpi: int) -> None:
    keys = ordered_keys(rows)
    labels: list[str] = []
    improvements: list[float] = []
    best_churns: list[float] = []

    for key in keys:
        values = dict(rows[key])
        if 0.0 not in values:
            continue
        best_churn, best_fid = min(rows[key], key=lambda item: item[1])
        labels.append(key)
        improvements.append(values[0.0] - best_fid)
        best_churns.append(best_churn)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(labels, improvements, color="#4c78a8")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Stage 2: best S_churn improvement over S_churn=0")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("FID-50k improvement")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, improvement, churn in zip(bars, improvements, best_churns):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 3 if y >= 0 else -3
        ax.annotate(
            f"S={churn:g}\n{improvement:.3g}",
            (bar.get_x() + bar.get_width() / 2, y),
            ha="center",
            va=va,
            textcoords="offset points",
            xytext=(0, offset),
            fontsize=8,
        )
    fig.subplots_adjust(bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Wrote {out}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary)
    if not rows:
        raise SystemExit(f"No plottable rows found in {args.summary}")

    plot_curve(rows, args.curve_out, args.dpi)
    plot_low_zoom(rows, args.zoom_out, args.dpi)
    plot_improvement(rows, args.improvement_out, args.dpi)


if __name__ == "__main__":
    main()
