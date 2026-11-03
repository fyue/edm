"""Plot the saved 30k-subset training evaluations without GPU computation."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


EXP_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=EXP_ROOT / "results/fid_summary.tsv")
    parser.add_argument("--outdir", type=Path, default=EXP_ROOT / "results")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: int(row["kimg"]))
    if not rows or any(int(row["num_images"]) != 50000 for row in rows):
        raise ValueError("Expected nonempty FID-50k evaluations.")
    if len({row["seeds"] for row in rows}) != 1:
        raise ValueError("All evaluations must use the same seed range.")
    x = [int(row["kimg"]) / 1000 for row in rows]
    y = [float(row["fid"]) for row in rows]
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 13, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.96)
    ax.plot(x, y, color="#0072B2", linewidth=2.5, marker="o", markersize=7)
    ax.set(xlabel="学習量（kimg）", ylabel="FID-50k",
           xlim=(5, max(x) + 5), ylim=(1.5, max(y) + 0.65), xticks=x)
    ax.set_xticklabels([f"{value:g}k" for value in x])
    ax.grid(axis="y", color="#E1E5E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#AAB2B9")
    args.outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output = args.outdir / f"training_fid50k.{suffix}"
        fig.savefig(output, dpi=200, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
