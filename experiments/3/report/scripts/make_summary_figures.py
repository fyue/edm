#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "assets"


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def collect() -> dict[str, dict[int, list[float]]]:
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in read_rows(ROOT / "1_120k-160k-lr-ema-sweep/results/summary.tsv"):
        label = {
            "lr1e-3-ema0.5-repro": "lr=1e-3, EMA=0.5",
            "lr5e-4-ema0.5": "lr=5e-4, EMA=0.5",
            "lr1e-3-ema1.0": "lr=1e-3, EMA=1.0",
        }[row["branch_id"]]
        data[label][int(row["kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "2_160k-200k-lr5e-4-ema0.5/results/summary.tsv"):
        data["lr=5e-4, EMA=0.5"][int(row["kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "4_fid50k_170k_lr5e-4_ema0.5/results/summary.tsv"):
        data["lr=5e-4, EMA=0.5"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "3_160k-170k-lr2e-4-ema0.5/results/summary.tsv"):
        data["lr=2e-4, EMA=0.5"][int(row["kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "5_170k-200k-lr2e-4-ema0.5/results/summary.tsv"):
        data["lr=2e-4, EMA=0.5"][int(row["kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "8_fid50k_167526_165018_lr2e-4_ema0.5/results/summary.tsv"):
        data["lr=2e-4, EMA=0.5"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/results/summary.tsv"):
        label = "lr=2e-4, EMA=0.5" if row["lr"] == "0.0002" else "lr=1e-4, EMA=0.5"
        data[label][int(row["snapshot_kimg"])].append(float(row["fid"]))

    exp10_rows = read_rows(ROOT / "10_160k-fid2.05-to-170k-lr2e-4-ema0.5/results/summary.tsv")
    exp10_by_kimg: dict[int, list[float]] = defaultdict(list)
    for row in exp10_rows:
        exp10_by_kimg[int(row["snapshot_kimg"])].append(float(row["fid"]))
    for kimg, values in exp10_by_kimg.items():
        if len(values) == 3:
            data["lr=2e-4 from FID2.05 160k"][kimg].extend(values)

    for row in read_rows(ROOT / "12_170k-to-180k-lr2e-4-ema0.5/results/summary.tsv"):
        data["lr=2e-4 from FID2.05 160k"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "13_170k-to-180k-lr1e-4-ema0.5/results/summary.tsv"):
        data["lr=1e-4 from best 170k"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "14_160k-fid2.05-to-175k-lr15e-4-ema0.5/results/summary.tsv"):
        data["lr=1.5e-3 from FID2.05 160k"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "16_160k-fid2.05-to-170k-lr15e-5-ema0.5/results/summary.tsv"):
        data["lr=1.5e-4 from FID2.05 160k"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "17_170k-to-175k-same-as-3-16-lr15e-5-ema0.5/results/summary.tsv"):
        data["lr=1.5e-4 from FID2.05 160k"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "18_160k-fid2.05-to-175k-lr1e-4-ema0.5/results/summary.tsv"):
        data["lr=1e-4 from FID2.05 160k"][int(row["snapshot_kimg"])].append(float(row["fid"]))

    for row in read_rows(ROOT / "11_fid50k_around170k_lr5e-4_ema0.5/results/summary.tsv"):
        kimg = int(row["snapshot_kimg"])
        if kimg == 170035:
            continue
        data["lr=5e-4, EMA=0.5"][kimg].append(float(row["fid"]))

    for row in read_rows(ROOT / "6_170k-180k-lr1e-4-ema0.5/results/summary.tsv"):
        if not row.get("kimg"):
            continue
        kimg = int(row["kimg"])
        label = "lr=2e-4, EMA=0.5" if kimg == 170000 else "lr=1e-4, EMA=0.5"
        data[label][kimg].append(float(row["fid"]))

    return data


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .tick{font-size:12px} .label{font-size:14px} .title{font-size:19px;font-weight:700}</style>',
    ]


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#333", width: float = 1, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle", weight: str = "400", rotate: str | None = None) -> str:
    rotate_attr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}"{rotate_attr}>{html.escape(value)}</text>'


def plot_160k_comparison(data: dict[str, dict[int, list[float]]]) -> None:
    width, height = 900, 520
    left, right, top, bottom = 95, 40, 70, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 1.88, 2.22

    def y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    labels = ["lr=1e-3, EMA=0.5", "lr=5e-4, EMA=0.5", "lr=1e-3, EMA=1.0"]
    colors = ["#4c78a8", "#59a14f", "#e15759"]
    stats = [mean_std(data[label][160000]) for label in labels]
    step = plot_w / len(labels)
    bar_w = 120

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "160k FID comparison: learning rate / EMA", 19, weight="700"))
    for tick in [1.9, 2.0, 2.1, 2.2]:
        yy = y(tick)
        parts.append(line(left, yy, width - right, yy, "#dddddd"))
        parts.append(text(left - 12, yy + 4, f"{tick:.1f}", 12, "end"))
    parts.append(line(left, top, left, top + plot_h))
    parts.append(line(left, top + plot_h, width - right, top + plot_h))
    parts.append(text(24, top + plot_h / 2, "FID-50k (lower is better)", 14, rotate="-90"))

    for i, (label, color, (mean, std)) in enumerate(zip(labels, colors, stats)):
        cx = left + step * (i + 0.5)
        yy = y(mean)
        h = top + plot_h - yy
        parts.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{yy:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" stroke="#333" stroke-width="1"/>')
        parts.append(line(cx, y(mean - std), cx, y(mean + std), "#333", 1.5))
        parts.append(line(cx - 14, y(mean - std), cx + 14, y(mean - std), "#333", 1.5))
        parts.append(line(cx - 14, y(mean + std), cx + 14, y(mean + std), "#333", 1.5))
        parts.append(text(cx, yy - 10, f"{mean:.3f}", 13, weight="700"))
        parts.append(text(cx, top + plot_h + 34, label, 13, rotate="-12"))
    parts.append("</svg>")
    (OUT / "fid_160k_comparison.svg").write_text("\n".join(parts) + "\n")


def plot_trajectory(data: dict[str, dict[int, list[float]]]) -> None:
    width, height = 980, 560
    left, right, top, bottom = 95, 205, 70, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = 158, 202
    ymin, ymax = 1.80, 2.75

    def x(k: float) -> float:
        return left + (k - xmin) / (xmax - xmin) * plot_w

    def y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    series = [
        ("lr=5e-4, EMA=0.5", "#59a14f"),
        ("lr=2e-4, EMA=0.5", "#f28e2b"),
        ("lr=2e-4 from FID2.05 160k", "#76b7b2"),
        ("lr=1.5e-4 from FID2.05 160k", "#ff9da7"),
        ("lr=1e-4 from FID2.05 160k", "#b07aa1"),
        ("lr=1e-4 from best 170k", "#edc948"),
        ("lr=1.5e-3 from FID2.05 160k", "#9c755f"),
        ("lr=1e-4, EMA=0.5", "#af7aa1"),
    ]

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "FID trajectory after 160k", 19, weight="700"))
    for tick in [1.9, 2.1, 2.3, 2.5, 2.7]:
        yy = y(tick)
        parts.append(line(left, yy, width - right, yy, "#dddddd"))
        parts.append(text(left - 12, yy + 4, f"{tick:.1f}", 12, "end"))
    for tick in [160, 170, 180, 190, 200]:
        xx = x(tick)
        parts.append(line(xx, top + plot_h, xx, top + plot_h + 6, "#333"))
        parts.append(text(xx, top + plot_h + 25, f"{tick}k", 12))
    parts.append(line(left, top, left, top + plot_h))
    parts.append(line(left, top + plot_h, width - right, top + plot_h))
    parts.append(text(24, top + plot_h / 2, "FID-50k (lower is better)", 14, rotate="-90"))
    parts.append(text(left + plot_w / 2, height - 16, "Training progress", 14))
    parts.append(line(x(170), top, x(170), top + plot_h, "#444", 1, "5,5"))
    parts.append(text(x(170) + 8, y(1.87), "best 170000", 12, "start", weight="700"))

    legend_y = top + 16
    for i, (label, color) in enumerate(series):
        ly = legend_y + i * 28
        parts.append(line(width - right + 28, ly, width - right + 66, ly, color, 3))
        parts.append(f'<circle cx="{width - right + 47}" cy="{ly}" r="4.5" fill="{color}"/>')
        parts.append(text(width - right + 76, ly + 5, label, 13, "start"))

        points: list[tuple[float, float, float]] = []
        for kimg, values in sorted(data[label].items()):
            if 160000 <= kimg <= 200000:
                mean, std = mean_std(values)
                points.append((kimg / 1000, mean, std))
        for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
            parts.append(line(x(x1), y(y1), x(x2), y(y2), color, 2.5))
        for k, mean, std in points:
            xx, yy = x(k), y(mean)
            parts.append(line(xx, y(mean - std), xx, y(mean + std), "#333", 1.2))
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.2" fill="{color}" stroke="#333" stroke-width="0.8"/>')
            parts.append(text(xx, yy - 10, f"{mean:.2f}", 11, weight="700"))

    parts.append("</svg>")
    (OUT / "fid_trajectory_after_160k.svg").write_text("\n".join(parts) + "\n")


def plot_progress_fid_by_condition(data: dict[str, dict[int, list[float]]]) -> None:
    width, height = 1080, 620
    left, right, top, bottom = 105, 245, 74, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = 158, 202
    ymin, ymax = 1.80, 2.75

    def x(k: float) -> float:
        return left + (k - xmin) / (xmax - xmin) * plot_w

    def y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    series = [
        ("lr=1e-3, EMA=0.5", "#4c78a8"),
        ("lr=1e-3, EMA=1.0", "#e15759"),
        ("lr=5e-4, EMA=0.5", "#59a14f"),
        ("lr=2e-4, EMA=0.5", "#f28e2b"),
        ("lr=2e-4 from FID2.05 160k", "#76b7b2"),
        ("lr=1.5e-4 from FID2.05 160k", "#ff9da7"),
        ("lr=1e-4 from FID2.05 160k", "#b07aa1"),
        ("lr=1e-4 from best 170k", "#edc948"),
        ("lr=1.5e-3 from FID2.05 160k", "#9c755f"),
        ("lr=1e-4, EMA=0.5", "#af7aa1"),
    ]

    parts = svg_header(width, height)
    parts.append(text(width / 2, 36, "Training progress vs FID by condition", 20, weight="700"))
    parts.append(text(width / 2, 58, "Each point is the mean of 3 FID-50k seed ranges; vertical bars show std.", 12))

    for tick in [1.9, 2.1, 2.3, 2.5, 2.7]:
        yy = y(tick)
        parts.append(line(left, yy, width - right, yy, "#dddddd"))
        parts.append(text(left - 12, yy + 4, f"{tick:.1f}", 12, "end"))
    for tick in [160, 170, 180, 190, 200]:
        xx = x(tick)
        parts.append(line(xx, top + plot_h, xx, top + plot_h + 6, "#333"))
        parts.append(text(xx, top + plot_h + 26, f"{tick}k", 12))
    parts.append(line(left, top, left, top + plot_h))
    parts.append(line(left, top + plot_h, width - right, top + plot_h))
    parts.append(text(26, top + plot_h / 2, "FID-50k (lower is better)", 14, rotate="-90"))
    parts.append(text(left + plot_w / 2, height - 19, "Training progress", 14))

    best_x = x(170)
    best_y = y(1.85124)
    parts.append(f'<rect x="{best_x - 45:.1f}" y="{best_y - 35:.1f}" width="90" height="52" rx="4" fill="#fff7e8" stroke="#f28e2b" stroke-width="1"/>')
    parts.append(text(best_x, best_y - 14, "best", 12, weight="700"))
    parts.append(text(best_x, best_y + 2, "170000 / 1.851", 12))

    legend_x = width - right + 30
    legend_y = top + 18
    for i, (label, color) in enumerate(series):
        ly = legend_y + i * 30
        parts.append(f'<circle cx="{legend_x:.1f}" cy="{ly:.1f}" r="5.4" fill="{color}" stroke="#333" stroke-width="0.8"/>')
        parts.append(text(legend_x + 14, ly + 5, label, 13, "start"))

    for label, color in series:
        points: list[tuple[float, float, float]] = []
        for kimg, values in sorted(data[label].items()):
            if 160000 <= kimg <= 200000:
                mean, std = mean_std(values)
                points.append((kimg / 1000, mean, std))

        for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
            parts.append(line(x(x1), y(y1), x(x2), y(y2), color, 2, "4,4"))

        for k, mean, std in points:
            xx, yy = x(k), y(mean)
            parts.append(line(xx, y(mean - std), xx, y(mean + std), "#333", 1.1))
            parts.append(line(xx - 7, y(mean - std), xx + 7, y(mean - std), "#333", 1.1))
            parts.append(line(xx - 7, y(mean + std), xx + 7, y(mean + std), "#333", 1.1))
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6.0" fill="{color}" stroke="#333" stroke-width="0.9"/>')
            label_y = yy - 12
            if label == "lr=5e-4, EMA=0.5" and k > 195:
                label_y = yy + 22
            if label == "lr=2e-4 from FID2.05 160k" and k == 170:
                continue
            parts.append(text(xx, label_y, f"{mean:.2f}", 11, weight="700"))

    callout_x = x(186)
    callout_y = y(2.02)
    parts.append(line(x(180), y(2.29788), callout_x, callout_y + 22, "#777", 1, "3,3"))
    parts.append(f'<rect x="{callout_x:.1f}" y="{callout_y:.1f}" width="250" height="55" rx="4" fill="#f7f7f7" stroke="#cccccc"/>')
    parts.append(text(callout_x + 12, callout_y + 20, "After 170k, FID worsens", 12, "start", weight="700"))
    parts.append(text(callout_x + 12, callout_y + 39, "even when training loss decreases.", 12, "start"))

    parts.append("</svg>")
    (OUT / "training_progress_fid_by_condition.svg").write_text("\n".join(parts) + "\n")


def plot_mean_fid_scatter() -> None:
    width, height = 980, 560
    left, right, top, bottom = 95, 230, 70, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = 158, 202
    ymin, ymax = 1.80, 2.75

    def x(k: float) -> float:
        return left + (k - xmin) / (xmax - xmin) * plot_w

    def y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    points = [
        ("lr=1e-3, EMA=0.5", 160, 2.04986, "#4c78a8"),
        ("lr=1e-3, EMA=1.0", 160, 2.16162, "#e15759"),
        ("lr=5e-4, EMA=0.5", 160, 2.00962, "#59a14f"),
        ("lr=5e-4, EMA=0.5", 165.018, 2.03798, "#59a14f"),
        ("lr=5e-4, EMA=0.5", 167.526, 2.03267, "#59a14f"),
        ("lr=5e-4, EMA=0.5", 170, 2.01113, "#59a14f"),
        ("lr=5e-4, EMA=0.5", 172.544, 2.03004, "#59a14f"),
        ("lr=5e-4, EMA=0.5", 180, 2.03250, "#59a14f"),
        ("lr=5e-4, EMA=0.5", 200, 2.29175, "#59a14f"),
        ("lr=2e-4 from FID2.05 160k", 162.509, 1.92373, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 165.018, 1.88942, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 167.526, 1.89796, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 170, 1.85124, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 172.509, 2.06431, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 175.018, 2.08354, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 177.526, 2.13972, "#76b7b2"),
        ("lr=2e-4 from FID2.05 160k", 180, 2.03784, "#76b7b2"),
        ("lr=1.5e-4 from FID2.05 160k", 162.509, 1.93085, "#ff9da7"),
        ("lr=1.5e-4 from FID2.05 160k", 165.018, 1.88729, "#ff9da7"),
        ("lr=1.5e-4 from FID2.05 160k", 167.526, 1.90077, "#ff9da7"),
        ("lr=1.5e-4 from FID2.05 160k", 170, 1.85363, "#ff9da7"),
        ("lr=1.5e-4 from FID2.05 160k", 172.509, 2.00938, "#ff9da7"),
        ("lr=1.5e-4 from FID2.05 160k", 175, 2.01434, "#ff9da7"),
        ("lr=1e-4 from FID2.05 160k", 162.509, 1.94893, "#b07aa1"),
        ("lr=1e-4 from FID2.05 160k", 165.018, 1.89113, "#b07aa1"),
        ("lr=1e-4 from FID2.05 160k", 167.526, 1.89994, "#b07aa1"),
        ("lr=1e-4 from FID2.05 160k", 170.035, 1.85841, "#b07aa1"),
        ("lr=1e-4 from FID2.05 160k", 172.544, 1.89526, "#b07aa1"),
        ("lr=1e-4 from FID2.05 160k", 175, 1.89012, "#b07aa1"),
        ("lr=1e-4 from best 170k", 172.509, 2.02557, "#edc948"),
        ("lr=1e-4 from best 170k", 175.018, 2.03641, "#edc948"),
        ("lr=1e-4 from best 170k", 177.526, 2.13456, "#edc948"),
        ("lr=1e-4 from best 170k", 180, 2.04304, "#edc948"),
        ("lr=1.5e-3 from FID2.05 160k", 162.509, 2.49590, "#9c755f"),
        ("lr=1.5e-3 from FID2.05 160k", 165.018, 2.54120, "#9c755f"),
        ("lr=1.5e-3 from FID2.05 160k", 167.526, 2.42753, "#9c755f"),
        ("lr=1.5e-3 from FID2.05 160k", 170.035, 2.34664, "#9c755f"),
        ("lr=1.5e-3 from FID2.05 160k", 172.544, 2.30579, "#9c755f"),
        ("lr=1.5e-3 from FID2.05 160k", 175, 2.30998, "#9c755f"),
        ("lr=2e-4, EMA=0.5", 165.018, 1.97630, "#f28e2b"),
        ("lr=2e-4, EMA=0.5", 167.526, 2.01374, "#f28e2b"),
        ("lr=2e-4, EMA=0.5", 170, 1.93936, "#f28e2b"),
        ("lr=2e-4, EMA=0.5", 172.509, 2.30210, "#f28e2b"),
        ("lr=2e-4, EMA=0.5", 175.018, 2.27436, "#f28e2b"),
        ("lr=2e-4, EMA=0.5", 180, 2.29788, "#f28e2b"),
        ("lr=1e-4, EMA=0.5", 172.509, 2.25384, "#af7aa1"),
        ("lr=1e-4, EMA=0.5", 175, 2.26373, "#af7aa1"),
        ("lr=1e-4, EMA=0.5", 180, 2.66206, "#af7aa1"),
    ]
    legend = [
        ("lr=1e-3, EMA=0.5", "#4c78a8"),
        ("lr=5e-4, EMA=0.5", "#59a14f"),
        ("lr=1e-3, EMA=1.0", "#e15759"),
        ("lr=2e-4, EMA=0.5", "#f28e2b"),
        ("lr=1e-4, EMA=0.5", "#af7aa1"),
        ("lr=2e-4 from FID2.05 160k", "#76b7b2"),
        ("lr=1e-4 from best 170k", "#edc948"),
        ("lr=1.5e-3 from FID2.05 160k", "#9c755f"),
        ("lr=1.5e-4 from FID2.05 160k", "#ff9da7"),
        ("lr=1e-4 from FID2.05 160k", "#b07aa1"),
    ]

    parts = svg_header(width, height)
    parts.append(text(width / 2, 36, "Training progress vs mean FID by condition", 20, weight="700"))
    parts.append(text(width / 2, 58, "Each point shows the mean FID over three seed ranges; lines include each branch's start checkpoint.", 12))

    for tick in [1.9, 2.1, 2.3, 2.5, 2.7]:
        yy = y(tick)
        parts.append(line(left, yy, width - right, yy, "#dddddd"))
        parts.append(text(left - 12, yy + 4, f"{tick:.1f}", 12, "end"))
    for tick in [160, 170, 180, 190, 200]:
        xx = x(tick)
        parts.append(line(xx, top + plot_h, xx, top + plot_h + 6, "#333"))
        parts.append(text(xx, top + plot_h + 25, f"{tick}k", 12))

    parts.append(line(left, top, left, top + plot_h))
    parts.append(line(left, top + plot_h, width - right, top + plot_h))
    parts.append(text(24, top + plot_h / 2, "Mean FID-50k", 14, rotate="-90"))
    parts.append(text(left + plot_w / 2, height - 16, "Training progress", 14))

    legend_x = width - right + 32
    legend_y = top + 20
    highlight_y = legend_y + (len(legend) - 2) * 30 - 17
    parts.append(
        f'<rect x="{legend_x - 18:.1f}" y="{highlight_y:.1f}" width="210" height="64" '
        'rx="4" fill="none" stroke="#2f6fdd" stroke-width="2"/>'
    )
    for i, (label, color) in enumerate(legend):
        ly = legend_y + i * 30
        parts.append(f'<circle cx="{legend_x:.1f}" cy="{ly:.1f}" r="6.2" fill="{color}" stroke="#333" stroke-width="0.8"/>')
        parts.append(text(legend_x + 16, ly + 5, label, 13, "start"))

    grouped: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for label, k, fid, color in points:
        grouped[label].append((k, fid, color))
    grouped["lr=2e-4, EMA=0.5"].append((160, 2.00962, "#f28e2b"))
    grouped["lr=2e-4 from FID2.05 160k"].append((160, 2.04986, "#76b7b2"))
    grouped["lr=1.5e-4 from FID2.05 160k"].append((160, 2.04986, "#ff9da7"))
    grouped["lr=1e-4 from FID2.05 160k"].append((160, 2.04986, "#b07aa1"))
    grouped["lr=1e-4 from best 170k"].append((170, 1.85124, "#edc948"))
    grouped["lr=1.5e-3 from FID2.05 160k"].append((160, 2.04986, "#9c755f"))
    grouped["lr=1e-4, EMA=0.5"].append((170, 1.93936, "#af7aa1"))

    for condition_points in grouped.values():
        condition_points.sort()
        for (k1, fid1, color), (k2, fid2, _) in zip(condition_points, condition_points[1:]):
            parts.append(line(x(k1), y(fid1), x(k2), y(fid2), color, 2.2))

    for _, k, fid, color in points:
        xx, yy = x(k), y(fid)
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="7.2" fill="{color}" stroke="#222" stroke-width="0.9"/>')
        parts.append(text(xx, yy - 11, f"{fid:.2f}", 11, weight="700"))

    parts.append("</svg>")
    (OUT / "training_progress_mean_fid_scatter.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = collect()
    plot_160k_comparison(data)
    plot_trajectory(data)
    plot_progress_fid_by_condition(data)
    plot_mean_fid_scatter()


if __name__ == "__main__":
    main()
