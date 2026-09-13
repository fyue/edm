#!/usr/bin/env python3
"""Measure 5-4/5-11 trajectory errors against subdivided-RK4 references."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dnnlib
import numpy as np
import torch

from generate import StackedRandomGenerator, parse_int_list


METRIC_FIELDS = ["seed", "class", "point", "sigma", "metric", "method", "relative_l2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Network pickle path or URL")
    parser.add_argument("--seeds", default="0-255", help="Seed list or range")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--rk-subdivisions", default="4,8")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seconds-per-nfe-image", type=float, default=0.00244)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def parse_subdivisions(value: str) -> tuple[int, int]:
    subdivisions = tuple(int(part) for part in value.split(","))
    if len(subdivisions) != 2 or subdivisions[0] >= subdivisions[1]:
        raise ValueError("--rk-subdivisions must contain two increasing integers")
    if any(value <= 0 for value in subdivisions):
        raise ValueError("RK subdivisions must be positive")
    return subdivisions


def estimate_nfe(steps: int, subdivisions: tuple[int, int]) -> int:
    positive_intervals = steps - 1
    reference_nfe = sum(positive_intervals * value * 4 + 1 for value in subdivisions)
    sampler_nfe = 2 * steps
    return reference_nfe + sampler_nfe


def build_sigma_steps(
    net: torch.nn.Module,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    indices = torch.arange(steps, dtype=torch.float64, device=device)
    sigma_steps = (
        sigma_max ** (1 / rho)
        + indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return net.round_sigma(sigma_steps)


def evaluate_ode(
    net: torch.nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    class_labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    denoised = net(x, sigma, class_labels).to(torch.float64)
    derivative = (x - denoised) / sigma
    return derivative, denoised


def variable_update_coefficients(
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    times: list[torch.Tensor],
) -> list[torch.Tensor]:
    coefficients = []
    one = torch.ones((), dtype=torch.float64, device=t_start.device)
    zero = torch.zeros((), dtype=torch.float64, device=t_start.device)
    for j, t_j in enumerate(times):
        polynomial = [one]
        for m, t_m in enumerate(times):
            if m == j:
                continue
            denominator = t_j - t_m
            updated = [zero for _ in range(len(polynomial) + 1)]
            for degree, value in enumerate(polynomial):
                updated[degree] = updated[degree] - value * t_m / denominator
                updated[degree + 1] = updated[degree + 1] + value / denominator
            polynomial = updated
        integral = zero
        for degree, value in enumerate(polynomial):
            integral = integral + value * (
                t_end ** (degree + 1) - t_start ** (degree + 1)
            ) / (degree + 1)
        coefficients.append(integral)
    return coefficients


def fixed_normalized_weights(order: int) -> list[float]:
    return {
        1: [1.0],
        2: [3 / 2, -1 / 2],
        3: [23 / 12, -16 / 12, 5 / 12],
        4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    }[order]


def run_ab_trajectory(
    net: torch.nn.Module,
    x_initial: torch.Tensor,
    class_labels: torch.Tensor | None,
    sigma_steps: torch.Tensor,
    variable: bool,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    states = [x_initial]
    denoised_states = []
    time_history: list[torch.Tensor] = []
    derivative_history: list[torch.Tensor] = []
    x = x_initial
    zero = torch.zeros_like(sigma_steps[:1])[0]

    for index, sigma_cur in enumerate(sigma_steps):
        sigma_next = sigma_steps[index + 1] if index + 1 < len(sigma_steps) else zero
        derivative, denoised = evaluate_ode(net, x, sigma_cur, class_labels)
        denoised_states.append(denoised)
        time_history.append(sigma_cur)
        derivative_history.append(derivative)
        time_history = time_history[-4:]
        derivative_history = derivative_history[-4:]

        if variable:
            coefficients = variable_update_coefficients(
                sigma_cur,
                sigma_next,
                time_history,
            )
            x_next = x
            for coefficient, old_derivative in zip(coefficients, derivative_history):
                x_next = x_next + coefficient * old_derivative
        else:
            step_size = sigma_next - sigma_cur
            weights = fixed_normalized_weights(len(derivative_history))
            combined = torch.zeros_like(derivative)
            for weight, old_derivative in zip(weights, reversed(derivative_history)):
                combined = combined + weight * old_derivative
            x_next = x + step_size * combined

        x = x_next
        states.append(x)

    return {
        "positive_states": states[:-1],
        "final": states[-1],
        "denoised": denoised_states,
    }


def run_reference_trajectory(
    net: torch.nn.Module,
    x_initial: torch.Tensor,
    class_labels: torch.Tensor | None,
    sigma_steps: torch.Tensor,
    subdivisions: int,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    states = [x_initial]
    outer_derivatives = []
    outer_denoised = []
    x = x_initial

    for sigma_cur, sigma_next in zip(sigma_steps[:-1], sigma_steps[1:]):
        outer_step = (sigma_next - sigma_cur) / subdivisions
        for substep in range(subdivisions):
            t_start = sigma_cur + substep * outer_step
            t_mid = t_start + 0.5 * outer_step
            t_end = t_start + outer_step

            k1, denoised = evaluate_ode(net, x, t_start, class_labels)
            if substep == 0:
                outer_derivatives.append(k1)
                outer_denoised.append(denoised)
            k2, _ = evaluate_ode(net, x + 0.5 * outer_step * k1, t_mid, class_labels)
            k3, _ = evaluate_ode(net, x + 0.5 * outer_step * k2, t_mid, class_labels)
            k4, _ = evaluate_ode(net, x + outer_step * k3, t_end, class_labels)
            x = x + outer_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(x)

    final_derivative, final_denoised = evaluate_ode(
        net,
        x,
        sigma_steps[-1],
        class_labels,
    )
    outer_derivatives.append(final_derivative)
    outer_denoised.append(final_denoised)
    return {
        "positive_states": states,
        "final": final_denoised,
        "derivatives": outer_derivatives,
        "denoised": outer_denoised,
    }


def relative_l2(
    value: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    difference = (value - reference).flatten(1).norm(dim=1)
    denominator = reference.flatten(1).norm(dim=1).clamp_min(epsilon)
    return difference / denominator


def local_predictions(
    reference: dict[str, list[torch.Tensor] | torch.Tensor],
    sigma_steps: torch.Tensor,
    variable: bool,
) -> list[torch.Tensor]:
    states = reference["positive_states"]
    derivatives = reference["derivatives"]
    assert isinstance(states, list)
    assert isinstance(derivatives, list)
    predictions = []
    zero = torch.zeros_like(sigma_steps[:1])[0]

    for index, sigma_cur in enumerate(sigma_steps):
        sigma_next = sigma_steps[index + 1] if index + 1 < len(sigma_steps) else zero
        start = max(0, index - 3)
        history_times = list(sigma_steps[start:index + 1])
        history_derivatives = derivatives[start:index + 1]
        if variable:
            coefficients = variable_update_coefficients(
                sigma_cur,
                sigma_next,
                history_times,
            )
            predicted = states[index]
            for coefficient, derivative in zip(coefficients, history_derivatives):
                predicted = predicted + coefficient * derivative
        else:
            step_size = sigma_next - sigma_cur
            weights = fixed_normalized_weights(len(history_derivatives))
            combined = torch.zeros_like(history_derivatives[-1])
            for weight, derivative in zip(weights, reversed(history_derivatives)):
                combined = combined + weight * derivative
            predicted = states[index] + step_size * combined
        predictions.append(predicted)
    return predictions


def add_metric_rows(
    rows: list[dict[str, str | int | float]],
    seeds: list[int],
    classes: list[int],
    point: int,
    sigma: float,
    metric: str,
    method: str,
    values: torch.Tensor,
) -> None:
    for seed, class_index, value in zip(seeds, classes, values.detach().cpu().tolist()):
        rows.append({
            "seed": seed,
            "class": class_index,
            "point": point,
            "sigma": f"{sigma:.12g}",
            "metric": metric,
            "method": method,
            "relative_l2": f"{value:.12g}",
        })


def compute_batch_rows(
    net: torch.nn.Module,
    seeds: list[int],
    sigma_steps: torch.Tensor,
    subdivisions: tuple[int, int],
    epsilon: float,
    device: torch.device,
) -> list[dict[str, str | int | float]]:
    random = StackedRandomGenerator(device, seeds)
    latents = random.randn(
        [len(seeds), net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    )
    class_labels = None
    classes = [-1] * len(seeds)
    if net.label_dim:
        class_indices = random.randint(net.label_dim, size=[len(seeds)], device=device)
        class_labels = torch.eye(net.label_dim, device=device)[class_indices]
        classes = class_indices.cpu().tolist()

    x_initial = latents.to(torch.float64) * sigma_steps[0]
    reference_coarse = run_reference_trajectory(
        net,
        x_initial,
        class_labels,
        sigma_steps,
        subdivisions[0],
    )
    reference_fine = run_reference_trajectory(
        net,
        x_initial,
        class_labels,
        sigma_steps,
        subdivisions[1],
    )
    variable = run_ab_trajectory(net, x_initial, class_labels, sigma_steps, variable=True)
    fixed = run_ab_trajectory(net, x_initial, class_labels, sigma_steps, variable=False)
    local_variable = local_predictions(reference_fine, sigma_steps, variable=True)
    local_fixed = local_predictions(reference_fine, sigma_steps, variable=False)

    coarse_states = reference_coarse["positive_states"]
    fine_states = reference_fine["positive_states"]
    variable_states = variable["positive_states"]
    fixed_states = fixed["positive_states"]
    fine_denoised = reference_fine["denoised"]
    variable_denoised = variable["denoised"]
    fixed_denoised = fixed["denoised"]
    assert isinstance(coarse_states, list)
    assert isinstance(fine_states, list)
    assert isinstance(variable_states, list)
    assert isinstance(fixed_states, list)
    assert isinstance(fine_denoised, list)
    assert isinstance(variable_denoised, list)
    assert isinstance(fixed_denoised, list)

    rows: list[dict[str, str | int | float]] = []
    for point, sigma_tensor in enumerate(sigma_steps):
        sigma = float(sigma_tensor)
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "reference_convergence",
            f"rk4-{subdivisions[0]}-vs-{subdivisions[1]}",
            relative_l2(coarse_states[point], fine_states[point], epsilon),
        )
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "global",
            "5-4-variable-ab",
            relative_l2(variable_states[point], fine_states[point], epsilon),
        )
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "global",
            "5-11-fixed-ab",
            relative_l2(fixed_states[point], fine_states[point], epsilon),
        )
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "denoised",
            "5-4-variable-ab",
            relative_l2(variable_denoised[point], fine_denoised[point], epsilon),
        )
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "denoised",
            "5-11-fixed-ab",
            relative_l2(fixed_denoised[point], fine_denoised[point], epsilon),
        )

        target = fine_states[point + 1] if point + 1 < len(fine_states) else reference_fine["final"]
        assert isinstance(target, torch.Tensor)
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "local",
            "5-4-variable-ab",
            relative_l2(local_variable[point], target, epsilon),
        )
        add_metric_rows(
            rows,
            seeds,
            classes,
            point,
            sigma,
            "local",
            "5-11-fixed-ab",
            relative_l2(local_fixed[point], target, epsilon),
        )

    final_point = len(sigma_steps)
    coarse_final = reference_coarse["final"]
    fine_final = reference_fine["final"]
    variable_final = variable["final"]
    fixed_final = fixed["final"]
    assert isinstance(coarse_final, torch.Tensor)
    assert isinstance(fine_final, torch.Tensor)
    assert isinstance(variable_final, torch.Tensor)
    assert isinstance(fixed_final, torch.Tensor)
    add_metric_rows(
        rows,
        seeds,
        classes,
        final_point,
        0.0,
        "reference_convergence",
        f"rk4-{subdivisions[0]}-vs-{subdivisions[1]}",
        relative_l2(coarse_final, fine_final, epsilon),
    )
    add_metric_rows(
        rows,
        seeds,
        classes,
        final_point,
        0.0,
        "global",
        "5-4-variable-ab",
        relative_l2(variable_final, fine_final, epsilon),
    )
    add_metric_rows(
        rows,
        seeds,
        classes,
        final_point,
        0.0,
        "global",
        "5-11-fixed-ab",
        relative_l2(fixed_final, fine_final, epsilon),
    )
    return rows


def read_completed_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row["seed"]) for row in csv.DictReader(f, delimiter="\t")}


def append_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path, expected_seeds: set[int]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            row for row in csv.DictReader(f, delimiter="\t")
            if int(row["seed"]) in expected_seeds
        ]


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[int, float, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["point"]),
            float(row["sigma"]),
            row["metric"],
            row["method"],
        )
        grouped[key].append(float(row["relative_l2"]))

    stats = []
    for (point, sigma, metric, method), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        stats.append({
            "point": point,
            "sigma": f"{sigma:.12g}",
            "metric": metric,
            "method": method,
            "n": len(values),
            "mean": f"{array.mean():.12g}",
            "median": f"{np.median(array):.12g}",
            "std": f"{array.std(ddof=1) if len(array) > 1 else 0.0:.12g}",
            "p90": f"{np.percentile(array, 90):.12g}",
        })
    return stats


def write_stats(path: Path, stats: list[dict[str, str | int | float]]) -> None:
    fields = ["point", "sigma", "metric", "method", "n", "mean", "median", "std", "p90"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(stats)


def plot_stats(path: Path, stats: list[dict[str, str | int | float]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/edm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("reference_convergence", "RK4 reference convergence"),
        ("global", "Global trajectory error"),
        ("local", "Local one-step error"),
        ("denoised", "Denoised-state error"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {
        "5-4-variable-ab": "#2563eb",
        "5-11-fixed-ab": "#dc2626",
        "rk4-4-vs-8": "#4b5563",
    }
    for axis, (metric, title) in zip(axes.flat, panels):
        methods = sorted({str(row["method"]) for row in stats if row["metric"] == metric})
        for method in methods:
            selected = [
                row for row in stats
                if row["metric"] == metric
                and row["method"] == method
                and float(row["sigma"]) > 0
            ]
            selected.sort(key=lambda row: int(row["point"]))
            sigma = np.asarray([float(row["sigma"]) for row in selected])
            mean = np.asarray([max(float(row["mean"]), 1e-16) for row in selected])
            p90 = np.asarray([max(float(row["p90"]), 1e-16) for row in selected])
            color = colors.get(method)
            axis.plot(sigma, mean, marker="o", markersize=3, color=color, label=f"{method} mean")
            axis.plot(sigma, p90, linestyle="--", linewidth=1, color=color, alpha=0.7, label=f"{method} p90")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(80, 0.002)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(title)
        axis.set_ylabel("relative L2 distance")
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("sigma (high to low, log scale)")
    fig.suptitle("Experiment 5-13: 5-4 vs 5-11 trajectory distances")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def find_stat(
    stats: list[dict[str, str | int | float]],
    point: int,
    metric: str,
    method: str,
) -> dict[str, str | int | float]:
    return next(
        row for row in stats
        if int(row["point"]) == point
        and row["metric"] == metric
        and row["method"] == method
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    subdivisions: tuple[int, int],
    stats: list[dict[str, str | int | float]],
    elapsed_seconds: float,
    nfe_per_image: int,
) -> None:
    final_point = args.steps
    reference = find_stat(
        stats,
        final_point,
        "reference_convergence",
        f"rk4-{subdivisions[0]}-vs-{subdivisions[1]}",
    )
    variable = find_stat(stats, final_point, "global", "5-4-variable-ab")
    fixed = find_stat(stats, final_point, "global", "5-11-fixed-ab")
    lines = [
        "# Experiment 5-13 trajectory-distance results",
        "",
        "## Condition",
        "",
        f"- seeds: `{args.seeds}`",
        f"- steps: `{args.steps}`",
        f"- sigma range: `{args.sigma_max}` to `{args.sigma_min}` with `rho={args.rho}`",
        f"- RK4 subdivisions: `{subdivisions[0]}` and `{subdivisions[1]}` per original interval",
        f"- estimated NFE per image: `{nfe_per_image}`",
        f"- measured elapsed time: `{elapsed_seconds / 60:.2f}` minutes",
        "",
        "## Final unquantized output distance",
        "",
        "| comparison | mean | median | std | p90 |",
        "|---|---:|---:|---:|---:|",
        f"| RK4-{subdivisions[0]} vs RK4-{subdivisions[1]} | {reference['mean']} | {reference['median']} | {reference['std']} | {reference['p90']} |",
        f"| 5-4 variable AB vs reference | {variable['mean']} | {variable['median']} | {variable['std']} | {variable['p90']} |",
        f"| 5-11 fixed AB vs reference | {fixed['mean']} | {fixed['median']} | {fixed['std']} | {fixed['p90']} |",
        "",
        "The reference final output is `D(x_ref(0.002), 0.002)`; the ODE is never evaluated at sigma=0.",
        "",
        "## Outputs",
        "",
        "- `per_seed_metrics.tsv`: scalar relative-L2 values for every seed and point",
        "- `step_stats.tsv`: mean, median, sample standard deviation, and 90th percentile",
        "- `trajectory_distances.svg`: log-sigma plots; sigma=0 final values are listed above",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_or_validate_config(path: Path, expected: dict[str, object]) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: current.get(key) for key in expected}
        if comparable != expected:
            raise SystemExit(
                f"Existing output config does not match this run: {path}\n"
                f"existing={comparable}\nrequested={expected}"
            )
        return
    path.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    subdivisions = parse_subdivisions(args.rk_subdivisions)
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    if not seeds:
        raise SystemExit("No seeds selected")

    nfe_per_image = estimate_nfe(args.steps, subdivisions)
    estimated_seconds = len(seeds) * nfe_per_image * args.seconds_per_nfe_image
    print(f"Seeds: {len(seeds)} ({args.seeds})")
    print(f"Estimated NFE/image: {nfe_per_image}")
    print(f"Estimated GPU time: {estimated_seconds / 60:.1f} minutes")
    if args.plan_only:
        return
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA is not available")

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.outdir / "per_seed_metrics.tsv"
    stats_path = args.outdir / "step_stats.tsv"
    plot_path = args.outdir / "trajectory_distances.svg"
    report_path = args.outdir / "README.md"
    config_path = args.outdir / "run_config.json"
    success_path = args.outdir / "_SUCCESS"
    expected_config = {
        "network": args.network,
        "seeds": args.seeds,
        "batch": args.batch,
        "steps": args.steps,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "rho": args.rho,
        "rk_subdivisions": list(subdivisions),
        "epsilon": args.epsilon,
        "nfe_per_image": nfe_per_image,
    }
    load_or_validate_config(config_path, expected_config)

    device = torch.device(args.device)
    print(f"Loading network from {args.network}")
    with dnnlib.util.open_url(args.network, verbose=True) as f:
        net = pickle.load(f)["ema"].eval().requires_grad_(False).to(device)
    sigma_steps = build_sigma_steps(
        net,
        args.steps,
        args.sigma_min,
        args.sigma_max,
        args.rho,
        device,
    )

    completed = read_completed_seeds(metrics_path)
    remaining = [seed for seed in seeds if seed not in completed]
    if completed:
        print(f"Resuming after {len(completed & set(seeds))} completed seeds")

    start_time = time.monotonic()
    with torch.no_grad():
        for batch_start in range(0, len(remaining), args.batch):
            batch_seeds = remaining[batch_start:batch_start + args.batch]
            batch_rows = compute_batch_rows(
                net,
                batch_seeds,
                sigma_steps,
                subdivisions,
                args.epsilon,
                device,
            )
            append_rows(metrics_path, batch_rows)
            done = min(batch_start + len(batch_seeds), len(remaining))
            print(f"Processed {done}/{len(remaining)} remaining seeds")

    elapsed_seconds = time.monotonic() - start_time
    all_rows = read_rows(metrics_path, set(seeds))
    expected_metrics_per_seed = args.steps * 7 + 3
    if len(all_rows) != len(seeds) * expected_metrics_per_seed:
        raise SystemExit(
            f"Incomplete metrics: got {len(all_rows)} rows, "
            f"expected {len(seeds) * expected_metrics_per_seed}"
        )
    stats = summarize_rows(all_rows)
    write_stats(stats_path, stats)
    plot_stats(plot_path, stats)
    write_report(
        report_path,
        args,
        subdivisions,
        stats,
        elapsed_seconds,
        nfe_per_image,
    )
    success_path.write_text("complete\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
