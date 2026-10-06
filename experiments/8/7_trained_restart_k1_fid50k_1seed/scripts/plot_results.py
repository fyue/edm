#!/usr/bin/env python3
"""Plot single-range FID and Restart improvement for Experiment 8-7."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "restart_improvements.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "restart_k1_fid50k_overview")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["complete"] == "1"]
    if not rows:
        print("No complete Restart/ODE pair yet; skipping plot")
        return
    for row in rows:
        if (row["num_images"], row["seeds"], row["nfe"]) != ("50000", "0-49999", "39"):
            raise ValueError(f"Unexpected plot protocol: {row}")
        for field in ("restart_fid", "same_nfe_ode_fid", "improvement_vs_same_nfe"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite plot value: {row}")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "edm-exp8-7-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    width = 0.34
    for offset, field, label, color in (
        (-width / 2, "same_nfe_ode_fid", "ODE, 20 steps (39 NFE)", "#0072B2"),
        (width / 2, "restart_fid", "Restart K=1 (39 NFE)", "#D55E00"),
    ):
        bars = axes[0].bar(x + offset, [float(row[field]) for row in rows], width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    axes[0].set_ylabel("FID-50k (lower is better)")
    axes[0].legend(fontsize=8)
    axes[0].margins(y=0.2)
    deltas = [float(row["improvement_vs_same_nfe"]) for row in rows]
    bars = axes[1].bar(x, deltas, width=0.55, color=["#009E73" if delta >= 0 else "#CC6677" for delta in deltas])
    axes[1].bar_label(bars, labels=[f"{delta:+.4f}" for delta in deltas], padding=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("FID(ODE) - FID(Restart)")
    axes[1].set_title("Positive favors Restart")
    axes[1].margins(y=0.25)
    labels = [{"exp3-best": "Exp. 3 best (170k)", "pretrained": "EDM pretrained"}.get(row["checkpoint"], row["checkpoint"]) for row in rows]
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Experiment 8-7: trained checkpoints, Restart K=1 vs. equal-NFE ODE")
    fig.text(0.5, 0.02, "CIFAR-10 conditional; seeds 0-49999; one FID evaluation per condition.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        target = args.output.with_suffix(suffix)
        fig.savefig(target, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
