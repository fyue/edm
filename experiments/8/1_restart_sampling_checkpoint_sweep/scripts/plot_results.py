#!/usr/bin/env python3
"""Plot Experiment 8-1 Restart sampling screening results."""

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
DEFAULT_INPUT = EXP_ROOT / "results" / "restart_fid_summary.tsv"
DEFAULT_OUTPUT_DIR = EXP_ROOT / "results"
CHECKPOINT_ORDER = ["5.5k", "20k", "120k", "200k"]
NFE_ORDER = [35, 39, 43, 55]
K_ORDER = [1, 2, 5]
K_COLORS = {1: "#0072B2", 2: "#E69F00", 5: "#009E73"}
ODE_COLOR = "#4D4D4D"
RESTART_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=1024)
    parser.add_argument("--seeds", default="0-1023")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_rows(path: Path, num_images: int, seeds: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("num_images") == str(num_images) and row.get("seeds") == seeds
        ]
    if not rows:
        raise ValueError(
            f"No rows for num_images={num_images}, seeds={seeds} in {path}"
        )
    for row in rows:
        try:
            fid = float(row["fid"])
            int(row["nfe"])
            int(row["restart_k"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result row: {row}") from exc
        if not math.isfinite(fid):
            raise ValueError(f"Non-finite FID in row: {row}")
    return rows


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    indexed: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (row["checkpoint"], row["method"], int(row["nfe"]))
        if key in indexed:
            raise ValueError(f"Duplicate protocol row: {key}")
        indexed[key] = float(row["fid"])
    return indexed


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


def plot_fid_curves(
    rows: list[dict[str, str]], output_dir: Path, num_images: int, seeds: str, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True)
    for ax, checkpoint in zip(axes.flat, CHECKPOINT_ORDER):
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        ode = sorted(
            (int(row["nfe"]), float(row["fid"]))
            for row in selected
            if row["method"] == "ode-heun"
        )
        restart = sorted(
            (int(row["nfe"]), float(row["fid"]))
            for row in selected
            if row["method"] == "restart"
        )
        if len(ode) != 4 or len(restart) != 3:
            raise ValueError(
                f"Expected 4 ODE and 3 Restart rows for {checkpoint}; "
                f"got {len(ode)} and {len(restart)}"
            )

        ax.plot(
            [nfe for nfe, _ in ode],
            [fid for _, fid in ode],
            color=ODE_COLOR,
            marker="o",
            linewidth=2.0,
            markersize=6,
            label="ODE (Heun)",
        )
        ax.plot(
            [nfe for nfe, _ in restart],
            [fid for _, fid in restart],
            color=RESTART_COLOR,
            marker="s",
            linewidth=2.0,
            markersize=6,
            label="Restart",
        )
        for points, offset, vertical_alignment in [
            (ode, 7, "bottom"),
            (restart, -9, "top"),
        ]:
            for nfe, fid in points:
                ax.annotate(
                    f"{fid:.2f}",
                    (nfe, fid),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=8,
                )
        ax.set_title(f"Checkpoint {checkpoint}")
        ax.set_xticks(NFE_ORDER)
        ax.set_ylabel(f"FID-{num_images:,} (lower is better)")
        ax.grid(True, alpha=0.25)
        ax.margins(y=0.14)

    for ax in axes[-1, :]:
        ax.set_xlabel("Number of function evaluations (NFE)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Experiment 8-1: Restart vs. ordinary ODE sampling", fontsize=14, y=0.995
    )
    fig.text(
        0.5,
        0.015,
        f"CIFAR-10 conditional; {num_images:,} images, seeds {seeds}. "
        "Screening result: small differences may be sampling noise.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.88), h_pad=2.0, w_pad=2.0)
    save_figure(fig, output_dir / "restart_fid_curves", dpi)
    plt.close(fig)


def plot_improvements(
    rows: list[dict[str, str]], output_dir: Path, num_images: int, seeds: str, dpi: int
) -> None:
    indexed = index_rows(rows)
    x = list(range(len(CHECKPOINT_ORDER)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    for restart_k in K_ORDER:
        restart_nfe = 35 + restart_k * 4
        restart_fids = [
            indexed[(checkpoint, "restart", restart_nfe)]
            for checkpoint in CHECKPOINT_ORDER
        ]
        main_improvements = [
            indexed[(checkpoint, "ode-heun", 35)] - restart_fid
            for checkpoint, restart_fid in zip(CHECKPOINT_ORDER, restart_fids)
        ]
        nfe_improvements = [
            indexed[(checkpoint, "ode-heun", restart_nfe)] - restart_fid
            for checkpoint, restart_fid in zip(CHECKPOINT_ORDER, restart_fids)
        ]
        for ax, values in zip(axes, [main_improvements, nfe_improvements]):
            ax.plot(
                x,
                values,
                color=K_COLORS[restart_k],
                marker="o",
                linewidth=2.0,
                markersize=6,
                label=f"K={restart_k} ({restart_nfe} NFE)",
            )

    titles = [
        "(a) Improvement over 18-step ODE",
        "(b) Improvement over equal-NFE ODE",
    ]
    for ax, title in zip(axes, titles):
        ax.axhline(0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks(x, CHECKPOINT_ORDER)
        ax.set_xlabel("Training checkpoint")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"FID-{num_images:,} reduction (positive favors Restart)")
    axes[1].legend(title="Restart setting", frameon=True)

    fig.suptitle("Experiment 8-1: Restart benefit over training progress", fontsize=14)
    fig.text(
        0.5,
        0.015,
        f"CIFAR-10 conditional; {num_images:,} images, seeds {seeds}. "
        "FID reduction = baseline FID - Restart FID.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.92), w_pad=2.5)
    save_figure(fig, output_dir / "restart_improvements", dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.num_images, args.seeds)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_fid_curves(rows, args.output_dir, args.num_images, args.seeds, args.dpi)
    plot_improvements(rows, args.output_dir, args.num_images, args.seeds, args.dpi)


if __name__ == "__main__":
    main()
