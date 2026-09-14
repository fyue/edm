#!/usr/bin/env python3
"""Plot the main Experiment 7-4 FID-50k comparisons."""

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
DEFAULT_RESULTS = EXP_ROOT / "results" / "fid50k_confirmation.tsv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "fid50k_confirmation_overview"

FM_NFE_CONDITIONS = ["fm-euler-nfe18", "fm-euler-nfe35"]
NFE35_CONDITIONS = ["fm-euler-nfe35", "edm-heun-nfe35"]
COLORS = {"FM": "#0072B2", "EDM": "#D55E00"}


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
        num_images = int(row["num_images"])
        fid = float(row["fid_50k"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid NFE, image count, or FID for {condition}") from exc
    if num_images != 50_000:
        raise ValueError(f"Expected 50,000 images for {condition}, got {num_images}")
    if not math.isfinite(fid):
        raise ValueError(f"Non-finite FID for {condition}: {fid}")
    return nfe, fid


def add_value_label(ax: plt.Axes, x: float, y: float) -> None:
    ax.annotate(
        f"{y:.3f}",
        (x, y),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )


def plot_fm_nfe_effect(
    ax: plt.Axes, results: dict[str, dict[str, str]]
) -> None:
    points = [numeric_result(results, condition) for condition in FM_NFE_CONDITIONS]
    nfes = [nfe for nfe, _ in points]
    fids = [fid for _, fid in points]

    ax.plot(
        nfes,
        fids,
        color=COLORS["FM"],
        marker="o",
        linewidth=2.4,
        markersize=7,
    )
    for nfe, fid in points:
        add_value_label(ax, nfe, fid)

    improvement = fids[0] - fids[1]
    relative = improvement / fids[0] * 100
    ax.text(
        0.5,
        0.08,
        f"FID reduction: {improvement:.3f} ({relative:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        color=COLORS["FM"],
        fontsize=10,
    )
    ax.set_title("(a) More NFE improves FM Euler")
    ax.set_xlabel("Number of function evaluations (NFE)")
    ax.set_ylabel("FID-50k (lower is better)")
    ax.set_xticks(nfes)
    ax.set_xlim(14, 39)
    ax.grid(True, alpha=0.25)


def plot_nfe35_comparison(
    ax: plt.Axes, results: dict[str, dict[str, str]]
) -> None:
    labels = ["FM\nEuler", "EDM\nHeun"]
    values = [numeric_result(results, condition)[1] for condition in NFE35_CONDITIONS]
    bars = ax.bar(labels, values, color=[COLORS["FM"], COLORS["EDM"]], width=0.58)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10, fontweight="bold")

    gap = values[0] - values[1]
    ax.text(
        0.5,
        0.90,
        f"EDM is lower by {gap:.3f} FID",
        transform=ax.transAxes,
        ha="center",
        color="dimgray",
        fontsize=10,
    )
    ax.set_title("(b) Model comparison at NFE 35")
    ax.set_xlabel("Sampler")
    ax.grid(axis="y", alpha=0.25)


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
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    plot_fm_nfe_effect(axes[0], results)
    plot_nfe35_comparison(axes[1], results)

    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max(8.1, ymax))

    fig.suptitle("Experiment 7-4: FID-50k confirmation", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "CIFAR-10 conditional, 20k kimg checkpoints, 50,000 generated images "
        "(seeds 0-49999).",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), w_pad=3.0)

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
