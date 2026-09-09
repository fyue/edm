#!/usr/bin/env python3
"""Compare variable- and fixed-step Adams-Bashforth coefficients."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_TSV = Path("experiments/5/report/assets/5-4_vs_5-11_ab_coefficients.tsv")
DEFAULT_MARKDOWN = Path("experiments/5/report/2026-09-01_ab_coefficients_5-4_vs_5-11.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def sigma_schedule(steps: int, sigma_min: float, sigma_max: float, rho: float) -> list[float]:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if not 0 < sigma_min < sigma_max:
        raise ValueError("require 0 < sigma_min < sigma_max")
    if rho <= 0:
        raise ValueError("rho must be positive")

    start = sigma_max ** (1 / rho)
    end = sigma_min ** (1 / rho)
    schedule = [
        (start + index / (steps - 1) * (end - start)) ** rho
        for index in range(steps)
    ]
    return [*schedule, 0.0]


def variable_update_coefficients(
    t_start: float,
    t_end: float,
    times: list[float],
) -> list[float]:
    """Integrate the Lagrange basis over one step, matching ab_sampler()."""
    coefficients = []
    for j, t_j in enumerate(times):
        polynomial = [1.0]
        for m, t_m in enumerate(times):
            if m == j:
                continue
            denominator = t_j - t_m
            updated = [0.0] * (len(polynomial) + 1)
            for degree, value in enumerate(polynomial):
                updated[degree] -= value * t_m / denominator
                updated[degree + 1] += value / denominator
            polynomial = updated

        integral = 0.0
        for degree, value in enumerate(polynomial):
            integral += value * (
                t_end ** (degree + 1) - t_start ** (degree + 1)
            ) / (degree + 1)
        coefficients.append(integral)
    return coefficients


def fixed_normalized_weights(order: int) -> list[float]:
    weights = {
        1: [1.0],
        2: [3 / 2, -1 / 2],
        3: [23 / 12, -16 / 12, 5 / 12],
        4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    }
    return weights[order]


def build_rows(schedule: list[float]) -> list[dict[str, float | int]]:
    rows = []
    time_history: list[float] = []
    for index, (sigma_cur, sigma_next) in enumerate(zip(schedule[:-1], schedule[1:])):
        time_history.append(sigma_cur)
        time_history = time_history[-4:]
        order = len(time_history)
        step_size = sigma_next - sigma_cur

        # ab_sampler() returns coefficients from the oldest derivative to the
        # current one. Reverse them so every output vector is [f_i, f_{i-1}, ...].
        variable_updates = list(reversed(variable_update_coefficients(
            sigma_cur,
            sigma_next,
            time_history,
        )))
        variable_weights = [value / step_size for value in variable_updates]
        fixed_weights = fixed_normalized_weights(order)
        fixed_updates = [step_size * value for value in fixed_weights]

        row: dict[str, float | int] = {
            "step": index + 1,
            "order": order,
            "sigma_cur": sigma_cur,
            "sigma_next": sigma_next,
            "delta_sigma": step_size,
        }
        for lag in range(4):
            row[f"variable_weight_f_i-{lag}"] = variable_weights[lag] if lag < order else 0.0
            row[f"fixed_weight_f_i-{lag}"] = fixed_weights[lag] if lag < order else 0.0
            row[f"variable_update_f_i-{lag}"] = variable_updates[lag] if lag < order else 0.0
            row[f"fixed_update_f_i-{lag}"] = fixed_updates[lag] if lag < order else 0.0
        row["max_abs_weight_diff"] = max(
            abs(float(row[f"variable_weight_f_i-{lag}"]) - float(row[f"fixed_weight_f_i-{lag}"]))
            for lag in range(4)
        )
        rows.append(row)
    return rows


def format_number(value: float) -> str:
    return f"{value:.9g}"


def write_tsv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: value if isinstance(value, int) else format_number(value)
                for key, value in row.items()
            })


def weight_vector(row: dict[str, float | int], prefix: str) -> str:
    order = int(row["order"])
    return "[" + ", ".join(
        f"{float(row[f'{prefix}_weight_f_i-{lag}']):.4f}"
        for lag in range(order)
    ) + "]"


def write_markdown(
    rows: list[dict[str, float | int]],
    path: Path,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    tsv_path: Path,
) -> None:
    lines = [
        "# 5-4と5-11のAdams-Bashforth係数比較",
        "",
        "この表は5-4（不等間隔に合わせた可変係数AB）と5-11（等間隔用の固定係数AB）で、",
        "各stepの更新に使われる係数を比較したものである。両実験はEDM rho schedule、",
        f"`steps={steps}`、`sigma_max={sigma_max:g}`、`sigma_min={sigma_min:g}`、`rho={rho:g}`を使う。",
        "EDM pretrainedの`round_sigma()`は入力をそのまま返すため、表のsigma値が実際の評価点になる。",
        "",
        "## 係数の読み方",
        "",
        "更新式を次の形で表す。",
        "",
        "```text",
        "x_{i+1} = x_i + Δσ_i (w0 f_i + w1 f_{i-1} + w2 f_{i-2} + w3 f_{i-3})",
        "Δσ_i = σ_{i+1} - σ_i < 0",
        "```",
        "",
        "下表の角括弧内は現在の微分から古い微分への順、すなわち",
        "`[f_i, f_{i-1}, f_{i-2}, f_{i-3}]`の正規化重み`w`である。",
        "実際に微分へ掛かる係数`Δσ_i w`はTSVに収録している。",
        "",
        "## Stepごとの正規化重み",
        "",
        "| step | order | sigma区間 | Δsigma | 5-4 可変係数 | 5-11 固定係数 | 最大差 |",
        "|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['step'])} | AB{int(row['order'])} | "
            f"{float(row['sigma_cur']):.6g} → {float(row['sigma_next']):.6g} | "
            f"{float(row['delta_sigma']):.6g} | "
            f"`{weight_vector(row, 'variable')}` | "
            f"`{weight_vector(row, 'fixed')}` | "
            f"{float(row['max_abs_weight_diff']):.4f} |"
        )

    first_ab4 = next(row for row in rows if int(row["order"]) == 4)
    last = rows[-1]
    lines.extend([
        "",
        "## 観察",
        "",
        "- 最初のEuler stepだけは両者が同一である。AB2以降はrho scheduleが不等間隔なため係数が異なる。",
        "- 5-11の固定AB4重みは全stepで`[2.2917, -2.4583, 1.5417, -0.3750]`のままである。",
        f"- 最初のAB4（step {int(first_ab4['step'])}）における5-4の重みは"
        f"`{weight_vector(first_ab4, 'variable')}`であり、すでに固定係数との差がある。",
        f"- 最終stepでは5-4が`{weight_vector(last, 'variable')}`、5-11が"
        f"`{weight_vector(last, 'fixed')}`となる。固定係数は低ノイズ側でも古い微分を大きく振動的に重み付けする。",
        "- したがって、両実験の差は単なる定数倍ではない。5-11は可変係数より強い外挿を全域、特に低ノイズ側で維持している。",
        "",
        "## FIDとの対応",
        "",
        "| 実験 | 係数 | steps / NFE | FID平均 |",
        "|---|---|---:|---:|",
        "| 5-4 | 可変係数 | 18 / 18 | 3.57748 |",
        "| 5-11 | 固定係数 | 18 / 18 | 1.99106 |",
        "",
        "固定係数は数値積分上の不等間隔補正を行っていないが、FIDは1.58642低かった。",
        "この結果は、固定係数が作る強い外挿バイアスが生成品質に有利に働いた可能性と整合する。",
        "ただし、係数比較だけではdenoiser予測誤差の補正を因果的に証明するものではない。",
        "",
        f"詳細な数値は[`{tsv_path.name}`](assets/{tsv_path.name})を参照する。",
        "",
        "## 再生成",
        "",
        "```bash",
        "python \\",
        "  experiments/5/report/scripts/compare_ab_coefficients.py",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    schedule = sigma_schedule(args.steps, args.sigma_min, args.sigma_max, args.rho)
    rows = build_rows(schedule)
    write_tsv(rows, args.tsv)
    write_markdown(
        rows,
        args.markdown,
        args.steps,
        args.sigma_min,
        args.sigma_max,
        args.rho,
        args.tsv,
    )
    print(f"Wrote {args.tsv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
