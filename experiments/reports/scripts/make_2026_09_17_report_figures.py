#!/usr/bin/env python3
from __future__ import annotations

import html
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "assets"


def text(
    x: float,
    y: float,
    value: str,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "400",
    rotate: int | None = None,
) -> str:
    rotate_attr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{rotate_attr}>{html.escape(value)}</text>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = "#333333",
    width: float = 1,
) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222;letter-spacing:0}</style>',
    ]


def add_bar_panel(
    parts: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    title_value: str,
    ylabel: str,
    values: list[tuple[str, float, str]],
    ymax: float,
    decimals: int,
) -> None:
    parts.append(text(x0 + width / 2, y0 - 22, title_value, 16, weight="700"))
    parts.append(line(x0, y0, x0, y0 + height))
    parts.append(line(x0, y0 + height, x0 + width, y0 + height))

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = ymax * ratio
        yy = y0 + height * (1 - ratio)
        parts.append(line(x0, yy, x0 + width, yy, "#dddddd"))
        parts.append(text(x0 - 9, yy + 4, f"{value:.{decimals}f}", 11, "end"))

    step = width / len(values)
    bar_width = min(92, step * 0.52)
    for index, (label, value, color) in enumerate(values):
        cx = x0 + step * (index + 0.5)
        yy = y0 + height * (1 - value / ymax)
        parts.append(
            f'<rect x="{cx - bar_width / 2:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
            f'height="{y0 + height - yy:.1f}" fill="{color}" stroke="#222" stroke-width="0.8"/>'
        )
        parts.append(text(cx, yy - 9, f"{value:.{decimals + 1}f}", 12, weight="700"))
        parts.append(text(cx, y0 + height + 24, label, 12, weight="700"))

    parts.append(text(x0 - 58, y0 + height / 2, ylabel, 12, rotate=-90))


def make_ab_comparison() -> None:
    width, height = 980, 460
    parts = header(width, height)
    parts.append(text(width / 2, 32, "Fixed vs variable Adams-Bashforth", 21, weight="700"))
    parts.append(text(width / 2, 54, "18 steps / 18 NFE; lower is better", 12))

    add_bar_panel(
        parts,
        82,
        102,
        350,
        270,
        "Endpoint error to RK4 reference",
        "Relative L2",
        [
            ("Variable AB", 0.0719539, "#e15759"),
            ("Fixed AB", 0.0228560, "#2f6fdd"),
        ],
        0.08,
        2,
    )
    add_bar_panel(
        parts,
        570,
        102,
        330,
        270,
        "FID-50k mean",
        "FID",
        [
            ("Variable AB", 3.57748, "#e15759"),
            ("Fixed AB", 1.99106, "#2f6fdd"),
        ],
        4.0,
        1,
    )
    parts.append(text(width / 2, 438, "The fixed coefficients are better on both trajectory error and FID.", 13))
    parts.append("</svg>")
    (OUT / "2026-09-17_ab_trajectory_fid.svg").write_text("\n".join(parts) + "\n")


def make_fm_endpoint_error() -> None:
    width, height = 900, 460
    parts = header(width, height)
    parts.append(text(width / 2, 32, "Integration error at 35 NFE", 21, weight="700"))
    parts.append(text(width / 2, 54, "Endpoint relative L2 to each model's RK4 reference; lower is better", 12))

    x0, y0, plot_width, plot_height = 90, 92, 740, 285
    ymax = 0.14
    parts.append(line(x0, y0, x0, y0 + plot_height))
    parts.append(line(x0, y0 + plot_height, x0 + plot_width, y0 + plot_height))
    for value in [0.00, 0.04, 0.08, 0.12]:
        yy = y0 + plot_height * (1 - value / ymax)
        parts.append(line(x0, yy, x0 + plot_width, yy, "#dddddd"))
        parts.append(text(x0 - 10, yy + 4, f"{value:.2f}", 11, "end"))

    groups = [
        ("FM", [("Euler", 0.122646, "#4c78a8"), ("Heun", 0.053771, "#f28e2b")]),
        ("EDM", [("Euler", 0.105352, "#4c78a8"), ("Heun", 0.033330, "#f28e2b")]),
    ]
    centers = [x0 + plot_width * 0.27, x0 + plot_width * 0.73]
    bar_width = 88
    gap = 18
    for center, (group_name, bars) in zip(centers, groups):
        for index, (label, value, color) in enumerate(bars):
            cx = center + (index - 0.5) * (bar_width + gap)
            yy = y0 + plot_height * (1 - value / ymax)
            parts.append(
                f'<rect x="{cx - bar_width / 2:.1f}" y="{yy:.1f}" width="{bar_width}" '
                f'height="{y0 + plot_height - yy:.1f}" fill="{color}" stroke="#222" stroke-width="0.8"/>'
            )
            parts.append(text(cx, yy - 9, f"{value:.3f}", 12, weight="700"))
            parts.append(text(cx, y0 + plot_height + 23, label, 12))
        parts.append(text(center, y0 + plot_height + 49, group_name, 14, weight="700"))

    parts.append(text(28, y0 + plot_height / 2, "Relative L2", 12, rotate=-90))
    parts.append("</svg>")
    (OUT / "2026-09-17_fm_endpoint_error.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_ab_comparison()
    make_fm_endpoint_error()


if __name__ == "__main__":
    main()
