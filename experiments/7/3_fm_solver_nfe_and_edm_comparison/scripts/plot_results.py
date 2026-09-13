#!/usr/bin/env python3
"""Plot the Experiment 7-3 FID-5k results."""

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


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = EXP_ROOT / "results" / "fm_solver_nfe_comparison.tsv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "fid5k_solver_nfe_overview"

FM_CONDITIONS = {
    "Euler": [
        "fm-euler-nfe5",
        "fm-euler-nfe9",
        "fm-euler-nfe18",
        "fm-euler-nfe19",
        "fm-euler-nfe35",
    ],
    "Heun": [
        "fm-heun-nfe5",
        "fm-heun-nfe9",
        "fm-heun-nfe19",
        "fm-heun-nfe35",
    ],
}

MODEL_COMPARISONS = [
    ("NFE 18\nEuler", "fm-euler-nfe18", "edm-euler-nfe18"),
    ("NFE 35\nHeun", "fm-heun-nfe35", "edm-heun-nfe35"),
]

COLORS = {"FM": "#0072B2", "EDM": "#D55E00"}
MARKERS = {"Euler": "o", "Heun": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; both PNG and PDF are written.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_results(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    results: dict[str, dict[str, str]] = {}
    for row in rows:
        condition = row.get("condition", "")
        if not condition:
            continue
        if condition in results:
            raise ValueError(f"Duplicate condition in {path}: {condition}")
        results[condition] = row
    return results


def numeric_result(
    results: dict[str, dict[str, str]], condition: str
) -> tuple[int, float]:
    if condition not in results:
        raise ValueError(f"Missing condition: {condition}")
    row = results[condition]
    try:
        nfe = int(row["nfe"])
        fid = float(row["fid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid NFE/FID for {condition}") from exc
    if not math.isfinite(fid):
        raise ValueError(f"Non-finite FID for {condition}: {fid}")
    return nfe, fid


def plot_fm_solver_sweep(
    ax: plt.Axes, results: dict[str, dict[str, str]]
) -> None:
    for solver, conditions in FM_CONDITIONS.items():
        points = [numeric_result(results, condition) for condition in conditions]
        ax.plot(
            [nfe for nfe, _ in points],
            [fid for _, fid in points],
            color=COLORS["FM"],
            linestyle="-" if solver == "Euler" else "--",
            marker=MARKERS[solver],
            linewidth=2.0,
            markersize=6,
            label=f"FM {solver}",
        )

    ax.set_title("(a) FM solver comparison")
    ax.set_xlabel("Number of function evaluations (NFE)")
    ax.set_ylabel("FID-5k (lower is better)")
    ax.set_xticks([5, 9, 18.5, 35], ["5", "9", "18/19", "35"])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)


def plot_model_comparison(
    ax: plt.Axes, results: dict[str, dict[str, str]]
) -> None:
    group_positions = list(range(len(MODEL_COMPARISONS)))
    width = 0.34
    fm_values = [numeric_result(results, fm_condition)[1] for _, fm_condition, _ in MODEL_COMPARISONS]
    edm_values = [numeric_result(results, edm_condition)[1] for _, _, edm_condition in MODEL_COMPARISONS]

    fm_bars = ax.bar(
        [position - width / 2 for position in group_positions],
        fm_values,
        width,
        color=COLORS["FM"],
        label="FM",
    )
    edm_bars = ax.bar(
        [position + width / 2 for position in group_positions],
        edm_values,
        width,
        color=COLORS["EDM"],
        label="EDM",
    )
    for bars in (fm_bars, edm_bars):
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    ax.set_title("(b) FM vs. EDM at matched solver and NFE")
    ax.set_ylabel("FID-5k (lower is better)")
    ax.set_xticks(group_positions, [label for label, _, _ in MODEL_COMPARISONS])
    ax.set_ylim(0, max(fm_values + edm_values) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)


def main() -> None:
    args = parse_args()
    results = load_results(args.results)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot_fm_solver_sweep(axes[0], results)
    plot_model_comparison(axes[1], results)
    fig.suptitle("Experiment 7-3: solver/NFE and model comparison", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "CIFAR-10 conditional, 20k kimg checkpoints, 5,000 generated images (seeds 0-4999).",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
