#!/usr/bin/env python3
from __future__ import annotations

import html
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "assets"
OUTPUT_DATE = "2026-09-01"

ROWS = [
    {
        "name": "EDM standard",
        "experiment": "reference",
        "steps": 18,
        "nfe": 35,
        "fid": 1.83906,
        "std": 0.01411,
        "color": "#4c78a8",
    },
    {
        "name": "PNDM 18 steps",
        "experiment": "5-1",
        "steps": 18,
        "nfe": 27,
        "fid": 1.99152,
        "std": 0.00708,
        "color": "#59a14f",
    },
    {
        "name": "S-PNDM 18 steps",
        "experiment": "5-2",
        "steps": 18,
        "nfe": 19,
        "fid": 3.14766,
        "std": 0.01782,
        "color": "#e15759",
    },
    {
        "name": "RK4 18 steps",
        "experiment": "5-3",
        "steps": 18,
        "nfe": 69,
        "fid": 1.92574,
        "std": 0.00804,
        "color": "#f28e2b",
    },
    {
        "name": "AB 18 steps",
        "experiment": "5-4",
        "steps": 18,
        "nfe": 18,
        "fid": 3.57748,
        "std": 0.01666,
        "color": "#9c755f",
    },
    {
        "name": "PNDM 26 steps",
        "experiment": "5-5",
        "steps": 26,
        "nfe": 35,
        "fid": 1.91047,
        "std": 0.00871,
        "color": "#76b7b2",
    },
    {
        "name": "PNDM linear",
        "experiment": "5-6",
        "steps": 18,
        "nfe": 27,
        "fid": 98.1024,
        "std": 0.18795,
        "color": "#b07aa1",
    },
    {
        "name": "AB 35 steps",
        "experiment": "5-7",
        "steps": 35,
        "nfe": 35,
        "fid": 2.12443,
        "std": 0.00789,
        "color": "#edc948",
    },
    {
        "name": "PNDM no-RK",
        "experiment": "5-8",
        "steps": 26,
        "nfe": 26,
        "fid": 1.91,
        "std": 0.00872,
        "color": "#ff9da7",
    },
    {
        "name": "PNDM no-RK 35 steps",
        "experiment": "5-9",
        "steps": 35,
        "nfe": 35,
        "fid": 1.89947,
        "std": 0.00806,
        "color": "#2f6fdd",
    },
    {
        "name": "S-PNDM 34 steps",
        "experiment": "5-10",
        "steps": 34,
        "nfe": 35,
        "fid": 2.21971,
        "std": 0.00691,
        "color": "#8c564b",
    },
    {
        "name": "PNDM no-RK 18 steps",
        "experiment": "5-11",
        "steps": 18,
        "nfe": 18,
        "fid": 1.99106,
        "std": 0.00635,
        "color": "#1b7f5a",
    },
    {
        "name": "AB 26 steps",
        "experiment": "5-12",
        "steps": 26,
        "nfe": 26,
        "fid": 2.45947,
        "std": 0.00721,
        "color": "#9467bd",
    },
]


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle", weight: str = "400", rotate: str | None = None) -> str:
    rotate_attr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}"{rotate_attr}>{html.escape(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#333", width: float = 1, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.note{fill:#555}</style>',
    ]


def plot_fid_bar() -> None:
    rows = [row for row in ROWS if row["fid"] < 10]
    width, height = 1280, 560
    left, right, top, bottom = 92, 35, 70, 132
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 1.75, 3.75
    bar_w = plot_w / len(rows) * 0.62

    def y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "Experiment 5 sampler comparison: mean FID", 20, weight="700"))
    parts.append(text(width / 2, 56, "CIFAR-10 conditional pretrained, FID-50k over 3 seed ranges, S_churn=0", 12, weight="400"))

    for tick in [2.0, 2.5, 3.0, 3.5]:
        yy = y(tick)
        parts.append(line(left, yy, width - right, yy, "#dddddd"))
        parts.append(text(left - 10, yy + 4, f"{tick:.1f}", 12, "end"))
    parts.append(line(left, top, left, top + plot_h))
    parts.append(line(left, top + plot_h, width - right, top + plot_h))
    parts.append(text(24, top + plot_h / 2, "FID-50k mean", 14, rotate="-90"))

    step = plot_w / len(rows)
    for i, row in enumerate(rows):
        cx = left + step * (i + 0.5)
        yy = y(row["fid"])
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
            f'height="{top + plot_h - yy:.1f}" fill="{row["color"]}" stroke="#222" stroke-width="0.8"/>'
        )
        parts.append(line(cx, y(row["fid"] - row["std"]), cx, y(row["fid"] + row["std"]), "#222", 1.1))
        parts.append(text(cx, yy - 9, f'{row["fid"]:.3f}', 11, weight="700"))
        parts.append(text(cx, top + plot_h + 28, f'{row["experiment"]}', 12, weight="700"))
        parts.append(text(cx, top + plot_h + 49, f'{row["name"]}', 12, rotate="-20"))
        parts.append(text(cx, top + plot_h + 88, f'NFE {row["nfe"]}', 11))

    parts.append(text(left, height - 18, "PNDM linear is omitted from this scale because its FID is 98.10.", 12, "start"))
    parts.append("</svg>")
    (OUT / f"{OUTPUT_DATE}_sampler_fid_comparison.svg").write_text("\n".join(parts) + "\n")


def plot_nfe_fid() -> None:
    rows = [row for row in ROWS if row["fid"] < 10]
    width, height = 1040, 540
    left, right, top, bottom = 85, 245, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = 15, 72
    ymin, ymax = 1.75, 3.75

    def x(v: float) -> float:
        return left + (v - xmin) / (xmax - xmin) * plot_w

    def y(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    parts = svg_header(width, height)
    parts.append(text(width / 2, 34, "NFE vs mean FID", 20, weight="700"))
    parts.append(text(width / 2, 56, "Lower-left is better. PNDM linear is out of range: FID=98.10 at NFE=27.", 12))

    for tick in [20, 35, 50, 65]:
        xx = x(tick)
        parts.append(line(xx, top + plot_h, xx, top + plot_h + 6, "#333"))
        parts.append(text(xx, top + plot_h + 25, str(tick), 12))
    for tick in [2.0, 2.5, 3.0, 3.5]:
        yy = y(tick)
        parts.append(line(left, yy, width - right, yy, "#dddddd"))
        parts.append(text(left - 10, yy + 4, f"{tick:.1f}", 12, "end"))
    parts.append(line(left, top, left, top + plot_h))
    parts.append(line(left, top + plot_h, width - right, top + plot_h))
    parts.append(text(left + plot_w / 2, height - 18, "NFE", 14))
    parts.append(text(24, top + plot_h / 2, "FID-50k mean", 14, rotate="-90"))

    label_positions = {
        "reference": (10, 18, "start"),
        "5-1": (10, -10, "start"),
        "5-5": (10, -13, "start"),
        "5-7": (10, 12, "start"),
        "5-8": (-10, 4, "end"),
        "5-9": (10, 4, "start"),
        "5-10": (10, -8, "start"),
        "5-11": (10, 4, "start"),
        "5-12": (-10, 4, "end"),
    }

    for row in rows:
        xx, yy = x(row["nfe"]), y(row["fid"])
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="7.0" fill="{row["color"]}" stroke="#222" stroke-width="0.9"/>')
        label = f'{row["experiment"]}: {row["fid"]:.2f}'
        default_position = (10, 4, "start") if row["nfe"] < 45 else (-10, 4, "end")
        dx, dy, anchor = label_positions.get(row["experiment"], default_position)
        parts.append(text(xx + dx, yy + dy, label, 11, anchor, weight="700"))

    legend_x, legend_y = width - right + 35, top + 8
    for i, row in enumerate(rows):
        ly = legend_y + i * 24
        parts.append(f'<circle cx="{legend_x:.1f}" cy="{ly:.1f}" r="5.0" fill="{row["color"]}" stroke="#222" stroke-width="0.8"/>')
        parts.append(text(legend_x + 12, ly + 4, f'{row["experiment"]} {row["name"]}', 11, "start"))

    parts.append("</svg>")
    (OUT / f"{OUTPUT_DATE}_sampler_nfe_vs_fid.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_fid_bar()
    plot_nfe_fid()


if __name__ == "__main__":
    main()
