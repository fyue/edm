#!/usr/bin/env python3
"""Create checkpoint-wise figures and wide tables for Experiment 8-4."""

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
from matplotlib.patches import Rectangle


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = EXP_ROOT / "results" / "restart_k_sweep.tsv"
DEFAULT_TABLE = EXP_ROOT / "results" / "fid_by_checkpoint_and_k.tsv"
DEFAULT_MARKDOWN = EXP_ROOT / "results" / "fid_by_checkpoint_and_k.md"
DEFAULT_PANELS = EXP_ROOT / "results" / "fid_vs_k_by_checkpoint"
DEFAULT_HEATMAP = EXP_ROOT / "results" / "fid_delta_vs_k0_heatmap"
EXPECTED_K = tuple(range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--panel-output", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--heatmap-output", type=Path, default=DEFAULT_HEATMAP)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_complete_sweeps(
    path: Path,
) -> list[tuple[str, int, dict[int, dict[str, str]]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("complete_sweep") == "1"
        ]

    grouped: dict[str, dict[int, dict[str, str]]] = {}
    training_kimg: dict[str, int] = {}
    for row in rows:
        checkpoint = row["checkpoint"]
        restart_k = int(row["restart_k"])
        fid = float(row["fid"])
        if restart_k not in EXPECTED_K or not math.isfinite(fid):
            raise ValueError(f"Invalid K or FID in row: {row}")
        if restart_k in grouped.setdefault(checkpoint, {}):
            raise ValueError(f"Duplicate checkpoint/K row: {(checkpoint, restart_k)}")
        grouped[checkpoint][restart_k] = row
        training_kimg[checkpoint] = int(row["training_kimg"])

    complete = []
    for checkpoint, by_k in grouped.items():
        if tuple(sorted(by_k)) != EXPECTED_K:
            continue
        complete.append((checkpoint, training_kimg[checkpoint], by_k))
    complete.sort(key=lambda item: item[1])
    if not complete:
        raise ValueError(f"No complete K=0..8 sweep in {path}")
    protocols = {
        (row["num_images"], row["seeds"])
        for _, _, by_k in complete
        for row in by_k.values()
    }
    if len(protocols) != 1:
        raise ValueError(f"Expected one FID protocol in {path}, got: {protocols}")
    return complete


def format_fid(value: float) -> str:
    return f"{value:.8g}"


def build_table_rows(
    sweeps: list[tuple[str, int, dict[int, dict[str, str]]]],
) -> list[dict[str, str]]:
    output = []
    for checkpoint, training_kimg, by_k in sweeps:
        ranked = sorted(
            ((float(by_k[restart_k]["fid"]), restart_k) for restart_k in EXPECTED_K)
        )
        best_fid, best_k = ranked[0]
        second_fid, second_k = ranked[1]
        k0_fid = float(by_k[0]["fid"])
        row = {
            "checkpoint": checkpoint,
            "training_kimg": str(training_kimg),
        }
        row.update(
            {
                f"fid_k{restart_k}": format_fid(float(by_k[restart_k]["fid"]))
                for restart_k in EXPECTED_K
            }
        )
        row.update(
            {
                "optimal_k": str(best_k),
                "optimal_fid": format_fid(best_fid),
                "second_best_k": str(second_k),
                "second_best_fid": format_fid(second_fid),
                "gap_to_second": format_fid(second_fid - best_fid),
                "improvement_vs_k0": format_fid(k0_fid - best_fid),
            }
        )
        output.append(row)
    return output


def write_tables(
    rows: list[dict[str, str]],
    table_output: Path,
    markdown_output: Path,
    num_images: int,
) -> None:
    fieldnames = [
        "checkpoint",
        "training_kimg",
        *[f"fid_k{restart_k}" for restart_k in EXPECTED_K],
        "optimal_k",
        "optimal_fid",
        "second_best_k",
        "second_best_fid",
        "gap_to_second",
        "improvement_vs_k0",
    ]
    table_output.parent.mkdir(parents=True, exist_ok=True)
    with table_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    header = [
        "Checkpoint",
        *[f"K={restart_k}" for restart_k in EXPECTED_K],
        "Best K",
        "Best FID",
        "2nd K",
        "Gap",
        "Gain vs K=0",
    ]
    lines = [
        "# Experiment 8-4: FID by checkpoint and Restart K",
        "",
        f"FID-{num_images:,} (lower is better). The minimum FID in each row is bold.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for row in rows:
        best_k = int(row["optimal_k"])
        fid_cells = []
        for restart_k in EXPECTED_K:
            value = f"{float(row[f'fid_k{restart_k}']):.5f}"
            fid_cells.append(f"**{value}**" if restart_k == best_k else value)
        cells = [
            row["checkpoint"],
            *fid_cells,
            row["optimal_k"],
            f"{float(row['optimal_fid']):.5f}",
            row["second_best_k"],
            f"{float(row['gap_to_second']):.5f}",
            f"{float(row['improvement_vs_k0']):.5f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Gap = second-best FID - best FID. "
            "Gain vs K=0 = FID(K=0) - best FID.",
            "",
        ]
    )
    markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {table_output}")
    print(f"Wrote {markdown_output}")


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_checkpoint_panels(
    sweeps: list[tuple[str, int, dict[int, dict[str, str]]]],
    output: Path,
    dpi: int,
) -> None:
    protocol = sweeps[0][2][0]
    num_images = int(protocol["num_images"])
    num_panels = len(sweeps)
    num_cols = min(4, num_panels)
    num_rows = math.ceil(num_panels / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.0 * num_cols, 3.25 * num_rows),
        squeeze=False,
    )

    for ax, (checkpoint, _, by_k) in zip(axes.flat, sweeps):
        restart_k = list(EXPECTED_K)
        fid = [float(by_k[value]["fid"]) for value in restart_k]
        best_k = min(EXPECTED_K, key=lambda value: (fid[value], value))
        best_fid = fid[best_k]
        k0_fid = fid[0]
        ax.plot(
            restart_k,
            fid,
            color="#0072B2",
            marker="o",
            linewidth=1.8,
            markersize=5,
            label="FID",
        )
        ax.axhline(
            k0_fid,
            color="#7F7F7F",
            linestyle="--",
            linewidth=1.1,
            label="K=0 FID",
        )
        ax.scatter(
            [best_k],
            [best_fid],
            color="#D55E00",
            marker="*",
            s=140,
            zorder=4,
            label="Minimum",
        )
        ax.annotate(
            f"K={best_k}, {best_fid:.4f}",
            (best_k, best_fid),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#A33A00",
        )
        ax.set_title(f"Checkpoint {checkpoint}")
        ax.set_xlabel("Restart K")
        ax.set_ylabel(f"FID-{num_images:,}")
        ax.set_xticks(EXPECTED_K)
        ax.grid(alpha=0.25)
        ax.margins(y=0.22)

    for ax in axes.flat[num_panels:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Experiment 8-4: FID-{num_images:,} versus Restart K at each checkpoint",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        f"CIFAR-10 conditional; {int(protocol['num_images']):,} images, "
        f"seeds {protocol['seeds']}. Each panel uses its own y-axis range.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90), h_pad=2.2, w_pad=1.6)
    save_figure(fig, output, dpi)


def plot_delta_heatmap(
    sweeps: list[tuple[str, int, dict[int, dict[str, str]]]],
    output: Path,
    dpi: int,
) -> None:
    num_images = int(sweeps[0][2][0]["num_images"])
    matrix = []
    best_k_values = []
    for _, _, by_k in sweeps:
        k0_fid = float(by_k[0]["fid"])
        values = [float(by_k[restart_k]["fid"]) - k0_fid for restart_k in EXPECTED_K]
        matrix.append(values)
        best_k_values.append(min(EXPECTED_K, key=lambda value: (values[value], value)))

    scale = max(abs(value) for row in matrix for value in row)
    scale = max(scale, 1e-6)
    fig_height = max(4.2, 0.55 * len(sweeps) + 2.2)
    fig, ax = plt.subplots(figsize=(10.8, fig_height))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    for row_index, values in enumerate(matrix):
        for restart_k, value in enumerate(values):
            color = "white" if abs(value) > 0.55 * scale else "black"
            ax.text(
                restart_k,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
        ax.add_patch(
            Rectangle(
                (best_k_values[row_index] - 0.48, row_index - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#00A000",
                linewidth=2.2,
            )
        )

    ax.set_xticks(EXPECTED_K, [f"K={value}" for value in EXPECTED_K])
    ax.set_yticks(
        range(len(sweeps)),
        [checkpoint for checkpoint, _, _ in sweeps],
    )
    ax.set_xlabel("Restart K")
    ax.set_ylabel("Training checkpoint")
    ax.set_title(
        f"FID-{num_images:,} change from K=0 ODE (green box: minimum FID)"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("FID(K) - FID(K=0); negative favors Restart")
    fig.tight_layout()
    save_figure(fig, output, dpi)


def main() -> None:
    args = parse_args()
    sweeps = load_complete_sweeps(args.sweep)
    rows = build_table_rows(sweeps)
    num_images = int(sweeps[0][2][0]["num_images"])
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    write_tables(rows, args.table_output, args.markdown_output, num_images)
    plot_checkpoint_panels(sweeps, args.panel_output, args.dpi)
    plot_delta_heatmap(sweeps, args.heatmap_output, args.dpi)


if __name__ == "__main__":
    main()
