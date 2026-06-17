#!/usr/bin/env python3
"""Plot Experiment 4 Stage 0 3x2 FID-1k vs S_churn curves as SVG.

This script intentionally uses only the Python standard library so it can run
inside the EDM conda environment without installing matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_SUMMARY = Path("experiments/4/stage0/3x2/results/stage0_summary.tsv")
DEFAULT_OUT = Path("experiments/4/stage0/3x2/results/stage0_fid_vs_s_churn.svg")
COLORS = {
    "10k": "#d55e00",
    "40k": "#0072b2",
    "120k": "#009e73",
    "pretrained": "#cc79a7",
}
ORDER = ["10k", "40k", "120k", "pretrained"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=920)
    parser.add_argument("--height", type=int, default=560)
    return parser.parse_args()


def load_rows(summary: Path) -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with summary.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                checkpoint = row["checkpoint"]
                churn = float(row["S_churn"])
                fid = float(row["fid_1k"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[checkpoint].append((churn, fid))
    return {key: sorted(value) for key, value in rows.items() if value}


def nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if math.isclose(vmin, vmax):
        pad = max(1.0, abs(vmin) * 0.05)
        vmin -= pad
        vmax += pad
    span = vmax - vmin
    raw_step = span / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    norm = raw_step / mag
    if norm <= 1:
        step = mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    cur = start
    while cur <= end + step * 0.5:
        ticks.append(cur)
        cur += step
    return ticks


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3g}"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def make_svg(rows: dict[str, list[tuple[float, float]]], width: int, height: int) -> str:
    margin_l, margin_r, margin_t, margin_b = 82, 210, 58, 76
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_points = [point for values in rows.values() for point in values]
    if not all_points:
        raise SystemExit("No plottable rows found in summary.")

    x_values = [x for x, _ in all_points]
    y_values = [y for _, y in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    if math.isclose(x_min, x_max):
        x_min -= 1
        x_max += 1
    y_pad = max(0.05, (y_max - y_min) * 0.15)
    y_min -= y_pad
    y_max += y_pad

    def xmap(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def ymap(y: float) -> float:
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    x_ticks = sorted(set(x_values))
    y_ticks = nice_ticks(y_min, y_max, count=6)
    ordered_keys = [key for key in ORDER if key in rows]
    ordered_keys.extend(sorted(key for key in rows if key not in ORDER))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_l}" y="30" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#222">Stage 0 3x2: FID-1k vs S_churn</text>',
        f'<text x="{margin_l}" y="50" font-family="Arial, sans-serif" font-size="13" fill="#666">samples=1024, seeds=0-1023, steps=18; FID-1k is for relative screening only.</text>',
    ]

    # Grid and axes.
    for tick in y_ticks:
        y = ymap(tick)
        parts.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#555">{fmt(tick)}</text>')
    for tick in x_ticks:
        x = xmap(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_t}" x2="{x:.2f}" y2="{margin_t + plot_h}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_t + plot_h + 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">{fmt(tick)}</text>')

    parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="1.5"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="1.5"/>')
    parts.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#222">S_churn</text>')
    parts.append(f'<text x="22" y="{margin_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {margin_t + plot_h / 2})" font-family="Arial, sans-serif" font-size="15" fill="#222">FID-1k</text>')

    for key in ordered_keys:
        points = rows[key]
        color = COLORS.get(key, "#333333")
        polyline = " ".join(f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in points)
        if len(points) > 1:
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            sx, sy = xmap(x), ymap(y)
            parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="4.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
            parts.append(f'<text x="{sx + 7:.2f}" y="{sy - 7:.2f}" font-family="Arial, sans-serif" font-size="11" fill="{color}">{y:.4g}</text>')

    # Legend.
    legend_x = margin_l + plot_w + 32
    legend_y = margin_t + 18
    parts.append(f'<text x="{legend_x}" y="{legend_y - 18}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#222">checkpoint</text>')
    for i, key in enumerate(ordered_keys):
        color = COLORS.get(key, "#333333")
        y = legend_y + i * 28
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="4.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
        parts.append(f'<text x="{legend_x + 40}" y="{y + 5}" font-family="Arial, sans-serif" font-size="13" fill="#333">{svg_escape(key)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary)
    svg = make_svg(rows, args.width, args.height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
