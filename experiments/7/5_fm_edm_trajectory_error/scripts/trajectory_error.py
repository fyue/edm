#!/usr/bin/env python3
"""Measure FM/EDM integration error and reference-trajectory geometry."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dnnlib
import numpy as np
import torch

from generate import (
    StackedRandomGenerator,
    ablation_sampler,
    edm_sampler,
    fm_euler_sampler,
    fm_heun_sampler,
    parse_int_list,
)


COARSE_SPECS = [
    ("euler", 5, 5),
    ("heun", 3, 5),
    ("euler", 9, 9),
    ("heun", 5, 9),
    ("euler", 18, 18),
    ("euler", 19, 19),
    ("heun", 10, 19),
    ("euler", 35, 35),
    ("heun", 18, 35),
]


def error_field(solver: str, nfe: int) -> str:
    return f"{solver}_nfe{nfe}_relative_l2"


ERROR_FIELDS = [error_field(solver, nfe) for solver, _, nfe in COARSE_SPECS]
PER_SEED_FIELDS = [
    "model",
    "seed",
    "class",
    "reference_coarse_steps",
    "reference_fine_steps",
    "reference_relative_l2",
    "path_length",
    "chord_length",
    "straightness_ratio",
    "max_chord_deviation_ratio",
    "mean_chord_deviation_ratio",
] + ERROR_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm-network", required=True)
    parser.add_argument("--edm-network", required=True)
    parser.add_argument("--seeds", default="0-63")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ref-coarse-steps", type=int, default=64)
    parser.add_argument("--ref-fine-steps", type=int, default=128)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds-per-nfe-image", type=float, default=0.00244)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def reference_nfe(model: str, steps: int) -> int:
    return 4 * steps + (1 if model == "edm" else 0)


def estimate_total_nfe_per_seed(coarse_steps: int, fine_steps: int) -> int:
    coarse_sampler_nfe = sum(nfe for _, _, nfe in COARSE_SPECS)
    return sum(
        reference_nfe(model, coarse_steps)
        + reference_nfe(model, fine_steps)
        + coarse_sampler_nfe
        for model in ("fm", "edm")
    )


def edm_sigma_and_derivative(
    progress: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = sigma_max ** (1 / rho)
    end = sigma_min ** (1 / rho)
    base = start + progress * (end - start)
    sigma = base ** rho
    derivative = rho * (end - start) * base ** (rho - 1)
    return sigma, derivative


def ode_derivative(
    model: str,
    net: torch.nn.Module,
    x: torch.Tensor,
    progress: torch.Tensor,
    class_labels: torch.Tensor | None,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> torch.Tensor:
    if model == "fm":
        time_batch = progress.to(torch.float64).repeat(x.shape[0])
        return net(x, time_batch, class_labels).to(torch.float64)
    if model == "edm":
        sigma, sigma_derivative = edm_sigma_and_derivative(
            progress,
            sigma_min,
            sigma_max,
            rho,
        )
        denoised = net(x, sigma, class_labels).to(torch.float64)
        return ((x - denoised) / sigma) * sigma_derivative
    raise ValueError(f"Unknown model: {model}")


def run_reference_trajectory(
    model: str,
    net: torch.nn.Module,
    x_initial: torch.Tensor,
    class_labels: torch.Tensor | None,
    steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    store_states: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Integrate one model ODE with fixed-step RK4 in normalized progress."""
    progress_steps = torch.linspace(
        0,
        1,
        steps + 1,
        dtype=torch.float64,
        device=x_initial.device,
    )
    x = x_initial
    states = [x] if store_states else []
    for progress_cur, progress_next in zip(progress_steps[:-1], progress_steps[1:]):
        h = progress_next - progress_cur
        progress_mid = progress_cur + 0.5 * h
        k1 = ode_derivative(
            model, net, x, progress_cur, class_labels,
            sigma_min, sigma_max, rho,
        )
        k2 = ode_derivative(
            model, net, x + 0.5 * h * k1, progress_mid, class_labels,
            sigma_min, sigma_max, rho,
        )
        k3 = ode_derivative(
            model, net, x + 0.5 * h * k2, progress_mid, class_labels,
            sigma_min, sigma_max, rho,
        )
        k4 = ode_derivative(
            model, net, x + h * k3, progress_next, class_labels,
            sigma_min, sigma_max, rho,
        )
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if store_states:
            states.append(x)

    if model == "edm":
        sigma = torch.as_tensor(sigma_min, dtype=torch.float64, device=x.device)
        x = net(x, sigma, class_labels).to(torch.float64)
        if store_states:
            states[-1] = x
    return x, states


def run_coarse_endpoint(
    model: str,
    solver: str,
    steps: int,
    net: torch.nn.Module,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None,
    random: StackedRandomGenerator,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    if model == "fm":
        sampler = fm_euler_sampler if solver == "euler" else fm_heun_sampler
        return sampler(
            net,
            latents,
            class_labels,
            randn_like=random.randn_like,
            num_steps=steps,
        )
    if model == "edm" and solver == "euler":
        return ablation_sampler(
            net,
            latents,
            class_labels,
            randn_like=random.randn_like,
            num_steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            solver="euler",
            discretization="edm",
            schedule="linear",
            scaling="none",
        )
    if model == "edm" and solver == "heun":
        return edm_sampler(
            net,
            latents,
            class_labels,
            randn_like=random.randn_like,
            num_steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            S_churn=0,
        )
    raise ValueError(f"Unsupported model/solver: {model}/{solver}")


def relative_l2(
    value: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    difference = (value - reference).flatten(1).norm(dim=1)
    denominator = reference.flatten(1).norm(dim=1).clamp_min(epsilon)
    return difference / denominator


def path_geometry(
    states: list[torch.Tensor],
    epsilon: float,
) -> dict[str, torch.Tensor]:
    if len(states) < 2:
        raise ValueError("At least two trajectory states are required")
    increments = [
        (next_state - state).flatten(1).norm(dim=1)
        for state, next_state in zip(states[:-1], states[1:])
    ]
    path_length = torch.stack(increments).sum(dim=0)
    chord = states[-1] - states[0]
    chord_length = chord.flatten(1).norm(dim=1).clamp_min(epsilon)
    deviations = []
    for index, state in enumerate(states):
        progress = index / (len(states) - 1)
        linear_state = states[0] + progress * chord
        deviations.append(
            (state - linear_state).flatten(1).norm(dim=1) / chord_length
        )
    deviation_stack = torch.stack(deviations)
    return {
        "path_length": path_length,
        "chord_length": chord_length,
        "straightness_ratio": path_length / chord_length,
        "max_chord_deviation_ratio": deviation_stack.max(dim=0).values,
        "mean_chord_deviation_ratio": deviation_stack.mean(dim=0),
    }


def initial_state(
    model: str,
    net: torch.nn.Module,
    latents: torch.Tensor,
    sigma_max: float,
) -> torch.Tensor:
    if model == "fm":
        return latents.to(torch.float64)
    rounded_max = net.round_sigma(
        torch.as_tensor(
            min(sigma_max, net.sigma_max),
            dtype=torch.float64,
            device=latents.device,
        )
    )
    return latents.to(torch.float64) * rounded_max


def compute_model_rows(
    model: str,
    net: torch.nn.Module,
    seeds: list[int],
    reference_coarse_steps: int,
    reference_fine_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    epsilon: float,
    device: torch.device,
) -> list[dict[str, str | int]]:
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

    effective_sigma_min = max(sigma_min, net.sigma_min) if model == "edm" else sigma_min
    effective_sigma_max = min(sigma_max, net.sigma_max) if model == "edm" else sigma_max
    x_initial = initial_state(model, net, latents, effective_sigma_max)
    coarse_reference, _ = run_reference_trajectory(
        model,
        net,
        x_initial,
        class_labels,
        reference_coarse_steps,
        effective_sigma_min,
        effective_sigma_max,
        rho,
        store_states=False,
    )
    fine_reference, fine_states = run_reference_trajectory(
        model,
        net,
        x_initial,
        class_labels,
        reference_fine_steps,
        effective_sigma_min,
        effective_sigma_max,
        rho,
        store_states=True,
    )
    values: dict[str, torch.Tensor] = {
        "reference_relative_l2": relative_l2(
            coarse_reference,
            fine_reference,
            epsilon,
        ),
        **path_geometry(fine_states, epsilon),
    }
    for solver, steps, nfe in COARSE_SPECS:
        endpoint = run_coarse_endpoint(
            model,
            solver,
            steps,
            net,
            latents,
            class_labels,
            random,
            effective_sigma_min,
            effective_sigma_max,
            rho,
        )
        values[error_field(solver, nfe)] = relative_l2(
            endpoint,
            fine_reference,
            epsilon,
        )

    cpu_values = {
        key: tensor.detach().cpu().tolist()
        for key, tensor in values.items()
    }
    rows = []
    for index, (seed, class_index) in enumerate(zip(seeds, classes)):
        row: dict[str, str | int] = {
            "model": model,
            "seed": seed,
            "class": class_index,
            "reference_coarse_steps": reference_coarse_steps,
            "reference_fine_steps": reference_fine_steps,
        }
        for key, batch_values in cpu_values.items():
            row[key] = f"{batch_values[index]:.12g}"
        rows.append(row)
    return rows


def read_rows(path: Path, expected: set[tuple[str, int]]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    latest: dict[tuple[str, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if any(not row.get(field) for field in PER_SEED_FIELDS):
                continue
            key = (row["model"], int(row["seed"]))
            if key in expected:
                latest[key] = row
    return [latest[key] for key in sorted(latest)]


def append_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_SEED_FIELDS, delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, str | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": f"{array.mean():.12g}",
        "median": f"{np.median(array):.12g}",
        "std": f"{array.std(ddof=1) if len(array) > 1 else 0.0:.12g}",
        "p90": f"{np.percentile(array, 90):.12g}",
    }


def endpoint_stats(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    stats = []
    for model in ("fm", "edm"):
        model_rows = [row for row in rows if row["model"] == model]
        reference = {
            "model": model,
            "solver": "rk4-reference",
            "steps": "coarse/fine",
            "nfe": "-",
            "metric": "reference_relative_l2",
            **summarize([float(row["reference_relative_l2"]) for row in model_rows]),
        }
        stats.append(reference)
        for solver, steps, nfe in COARSE_SPECS:
            field = error_field(solver, nfe)
            stats.append({
                "model": model,
                "solver": solver,
                "steps": steps,
                "nfe": nfe,
                "metric": "endpoint_relative_l2",
                **summarize([float(row[field]) for row in model_rows]),
            })
    return stats


def geometry_stats(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    fields = [
        "path_length",
        "chord_length",
        "straightness_ratio",
        "max_chord_deviation_ratio",
        "mean_chord_deviation_ratio",
    ]
    stats = []
    for model in ("fm", "edm"):
        model_rows = [row for row in rows if row["model"] == model]
        for metric in fields:
            stats.append({
                "model": model,
                "metric": metric,
                **summarize([float(row[metric]) for row in model_rows]),
            })
    return stats


def write_tsv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_endpoint_errors(
    path: Path,
    stats: list[dict[str, str | int]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/edm-exp7-5-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {"euler": "#2563eb", "heun": "#dc2626"}
    for axis, model in zip(axes, ("fm", "edm")):
        for solver in ("euler", "heun"):
            selected = [
                row for row in stats
                if row["model"] == model and row["solver"] == solver
            ]
            selected.sort(key=lambda row: int(row["nfe"]))
            axis.plot(
                [int(row["nfe"]) for row in selected],
                [float(row["mean"]) for row in selected],
                marker="o",
                color=colors[solver],
                label=solver.capitalize(),
            )
        reference = next(
            row for row in stats
            if row["model"] == model and row["solver"] == "rk4-reference"
        )
        axis.axhline(
            float(reference["mean"]),
            color="#4b5563",
            linestyle="--",
            label="RK4 coarse vs fine",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([5, 9, 18, 19, 35], labels=[5, 9, 18, 19, 35])
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(model.upper())
        axis.set_xlabel("NFE")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean endpoint relative L2")
    fig.suptitle("Experiment 7-5: integration error vs NFE")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_path_geometry(
    path: Path,
    stats: list[dict[str, str | int]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/edm-exp7-5-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("straightness_ratio", "Path / chord length"),
        ("max_chord_deviation_ratio", "Max chord deviation"),
        ("mean_chord_deviation_ratio", "Mean chord deviation"),
    ]
    colors = {"fm": "#2563eb", "edm": "#dc2626"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (metric, title) in zip(axes, metrics):
        selected = [row for row in stats if row["metric"] == metric]
        selected.sort(key=lambda row: str(row["model"]))
        labels = [str(row["model"]).upper() for row in selected]
        means = [float(row["mean"]) for row in selected]
        axis.bar(
            labels,
            means,
            color=[colors[str(row["model"])] for row in selected],
        )
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Experiment 7-5: fine-reference path geometry")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


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
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    endpoint: list[dict[str, str | int]],
    geometry: list[dict[str, str | int]],
    elapsed_seconds: float,
    nfe_per_seed: int,
) -> None:
    lines = [
        "# Experiment 7-5 results",
        "",
        "## Conditions",
        "",
        f"- seeds: `{args.seeds}`",
        f"- RK4 reference steps: `{args.ref_coarse_steps}` / `{args.ref_fine_steps}`",
        f"- estimated total NFE per seed across FM and EDM: `{nfe_per_seed}`",
        f"- elapsed in this invocation: `{elapsed_seconds / 60:.2f}` minutes",
        "",
        "## Endpoint relative L2",
        "",
        "| model | solver | steps | NFE | mean | median | std | p90 |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in endpoint:
        lines.append(
            f"| {row['model']} | {row['solver']} | {row['steps']} | {row['nfe']} | "
            f"{row['mean']} | {row['median']} | {row['std']} | {row['p90']} |"
        )
    lines.extend([
        "",
        "## Reference-path geometry",
        "",
        "| model | metric | mean | median | std | p90 |",
        "|:--|:--|--:|--:|--:|--:|",
    ])
    for row in geometry:
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['mean']} | "
            f"{row['median']} | {row['std']} | {row['p90']} |"
        )
    lines.extend([
        "",
        "The EDM reference integrates only to sigma_min and then applies the terminal denoise; "
        "the network is never evaluated at sigma=0.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_network(model: str, net: torch.nn.Module) -> None:
    prediction_type = getattr(net, "prediction_type", "denoiser")
    if model == "fm" and prediction_type != "velocity":
        raise SystemExit("FM checkpoint does not declare prediction_type=velocity")
    if model == "edm" and prediction_type == "velocity":
        raise SystemExit("EDM checkpoint unexpectedly declares prediction_type=velocity")


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("No seeds selected")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.ref_coarse_steps < 2:
        raise SystemExit("--ref-coarse-steps must be at least 2")
    if args.ref_fine_steps <= args.ref_coarse_steps:
        raise SystemExit("--ref-fine-steps must exceed --ref-coarse-steps")
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        raise SystemExit("Require 0 < sigma_min < sigma_max")

    nfe_per_seed = estimate_total_nfe_per_seed(
        args.ref_coarse_steps,
        args.ref_fine_steps,
    )
    estimated_seconds = len(seeds) * nfe_per_seed * args.seconds_per_nfe_image
    print(f"Seeds: {len(seeds)} ({args.seeds})")
    print(f"Coarse samplers per model: {len(COARSE_SPECS)}")
    print(f"Estimated total NFE/seed across both models: {nfe_per_seed}")
    print(f"Estimated GPU time: {estimated_seconds / 60:.1f} minutes")
    if args.plan_only:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.outdir / "per_seed_metrics.tsv"
    endpoint_path = args.outdir / "endpoint_error_stats.tsv"
    geometry_path = args.outdir / "path_geometry_stats.tsv"
    endpoint_plot_path = args.outdir / "endpoint_errors.svg"
    geometry_plot_path = args.outdir / "path_geometry.svg"
    report_path = args.outdir / "README.md"
    config_path = args.outdir / "run_config.json"
    success_path = args.outdir / "_SUCCESS"
    expected_config = {
        "fm_network": args.fm_network,
        "edm_network": args.edm_network,
        "seeds": args.seeds,
        "batch": args.batch,
        "reference_coarse_steps": args.ref_coarse_steps,
        "reference_fine_steps": args.ref_fine_steps,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "rho": args.rho,
        "epsilon": args.epsilon,
        "coarse_specs": [list(spec) for spec in COARSE_SPECS],
        "estimated_total_nfe_per_seed": nfe_per_seed,
    }
    load_or_validate_config(config_path, expected_config)

    device = torch.device(args.device)
    expected_keys = {
        (model, seed)
        for model in ("fm", "edm")
        for seed in seeds
    }
    start_time = time.monotonic()
    for model, network_path in (
        ("fm", args.fm_network),
        ("edm", args.edm_network),
    ):
        completed = {
            (row["model"], int(row["seed"]))
            for row in read_rows(metrics_path, expected_keys)
        }
        remaining = [seed for seed in seeds if (model, seed) not in completed]
        if not remaining:
            print(f"[{model}] all {len(seeds)} seeds already complete")
            continue
        print(f"[{model}] loading network from {network_path}")
        with dnnlib.util.open_url(network_path, verbose=True) as file:
            net = pickle.load(file)["ema"].eval().requires_grad_(False).to(device)
        validate_network(model, net)
        with torch.no_grad():
            for batch_start in range(0, len(remaining), args.batch):
                batch_seeds = remaining[batch_start:batch_start + args.batch]
                rows = compute_model_rows(
                    model,
                    net,
                    batch_seeds,
                    args.ref_coarse_steps,
                    args.ref_fine_steps,
                    args.sigma_min,
                    args.sigma_max,
                    args.rho,
                    args.epsilon,
                    device,
                )
                append_rows(metrics_path, rows)
                done = min(batch_start + len(batch_seeds), len(remaining))
                print(f"[{model}] processed {done}/{len(remaining)} remaining seeds")
        del net
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_seconds = time.monotonic() - start_time
    all_rows = read_rows(metrics_path, expected_keys)
    if len(all_rows) != len(expected_keys):
        raise SystemExit(
            f"Incomplete metrics: got {len(all_rows)} model/seed rows, "
            f"expected {len(expected_keys)}"
        )
    endpoint = endpoint_stats(all_rows)
    geometry = geometry_stats(all_rows)
    write_tsv(endpoint_path, endpoint)
    write_tsv(geometry_path, geometry)
    plot_endpoint_errors(endpoint_plot_path, endpoint)
    plot_path_geometry(geometry_plot_path, geometry)
    write_report(
        report_path,
        args,
        endpoint,
        geometry,
        elapsed_seconds,
        nfe_per_seed,
    )
    success_path.write_text("complete\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {endpoint_path}")
    print(f"Wrote {geometry_path}")
    print(f"Wrote {endpoint_plot_path}")
    print(f"Wrote {geometry_plot_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
