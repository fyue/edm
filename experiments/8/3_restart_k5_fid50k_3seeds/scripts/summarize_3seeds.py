#!/usr/bin/env python3
"""Aggregate paired Restart improvements over independent seed ranges."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXP_ROOT / "results" / "restart_improvements.tsv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "restart_k5_fid50k_3seed_stats.tsv"
DEFAULT_SEEDS = "0-49999 50000-99999 100000-149999"
CHECKPOINT_ORDER = ["5.5k", "20k", "120k", "200k"]
NUMERIC_FIELDS = [
    "restart_fid",
    "same_main_ode_fid",
    "improvement_vs_same_main",
    "same_nfe_ode_fid",
    "improvement_vs_same_nfe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument(
        "--expected-seeds",
        default=DEFAULT_SEEDS,
        help="Space-separated seed ranges expected for a complete result.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        sum((value - center) ** 2 for value in values) / (len(values) - 1)
    )


def fmt(value: float) -> str:
    return f"{value:.8g}"


def parse_numeric_row(row: dict[str, str]) -> dict[str, float] | None:
    parsed: dict[str, float] = {}
    try:
        for field in NUMERIC_FIELDS:
            parsed[field] = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in parsed.values()):
        return None
    return parsed


def read_groups(
    path: Path, num_images: int, expected_seeds: set[str]
) -> dict[tuple[str, str, str, str, str], dict[str, dict[str, float]]]:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    groups: dict[
        tuple[str, str, str, str, str], dict[str, dict[str, float]]
    ] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("num_images") != str(num_images):
                continue
            seeds = row.get("seeds", "")
            if seeds not in expected_seeds:
                continue
            values = parse_numeric_row(row)
            if values is None:
                continue
            key = (
                row["condition"],
                row["checkpoint"],
                row["restart_k"],
                row["nfe"],
                row["num_images"],
            )
            if seeds in groups[key]:
                raise ValueError(f"Duplicate condition/seed result: {key}, {seeds}")
            groups[key][seeds] = values
    return groups


def group_order(key: tuple[str, str, str, str, str]) -> tuple[int, str]:
    checkpoint = key[1]
    try:
        rank = CHECKPOINT_ORDER.index(checkpoint)
    except ValueError:
        rank = len(CHECKPOINT_ORDER)
    return rank, checkpoint


def main() -> None:
    args = parse_args()
    expected_order = args.expected_seeds.split()
    if not expected_order or len(expected_order) != len(set(expected_order)):
        raise SystemExit("--expected-seeds must contain unique seed ranges")
    expected_set = set(expected_order)
    groups = read_groups(args.input, args.num_images, expected_set)

    fieldnames = [
        "condition",
        "checkpoint",
        "restart_k",
        "nfe",
        "num_images",
        "n_seed_ranges",
        "restart_fid_mean",
        "restart_fid_std",
        "same_main_ode_fid_mean",
        "same_main_ode_fid_std",
        "same_nfe_ode_fid_mean",
        "same_nfe_ode_fid_std",
        "improvement_vs_same_main_mean",
        "improvement_vs_same_main_std",
        "improvement_vs_same_main_min",
        "improvement_vs_same_main_max",
        "improvement_vs_same_nfe_mean",
        "improvement_vs_same_nfe_std",
        "improvement_vs_same_nfe_min",
        "improvement_vs_same_nfe_max",
        "seed_values",
        "complete",
    ]

    output_rows: list[dict[str, str | int]] = []
    for key in sorted(groups, key=group_order):
        condition, checkpoint, restart_k, nfe, num_images = key
        by_seed = groups[key]
        ordered_seeds = [seed for seed in expected_order if seed in by_seed]
        series = {
            field: [by_seed[seed][field] for seed in ordered_seeds]
            for field in NUMERIC_FIELDS
        }
        main_improvements = series["improvement_vs_same_main"]
        nfe_improvements = series["improvement_vs_same_nfe"]
        output_rows.append(
            {
                "condition": condition,
                "checkpoint": checkpoint,
                "restart_k": restart_k,
                "nfe": nfe,
                "num_images": num_images,
                "n_seed_ranges": len(ordered_seeds),
                "restart_fid_mean": fmt(mean(series["restart_fid"])),
                "restart_fid_std": fmt(sample_std(series["restart_fid"])),
                "same_main_ode_fid_mean": fmt(mean(series["same_main_ode_fid"])),
                "same_main_ode_fid_std": fmt(sample_std(series["same_main_ode_fid"])),
                "same_nfe_ode_fid_mean": fmt(mean(series["same_nfe_ode_fid"])),
                "same_nfe_ode_fid_std": fmt(sample_std(series["same_nfe_ode_fid"])),
                "improvement_vs_same_main_mean": fmt(mean(main_improvements)),
                "improvement_vs_same_main_std": fmt(sample_std(main_improvements)),
                "improvement_vs_same_main_min": fmt(min(main_improvements)),
                "improvement_vs_same_main_max": fmt(max(main_improvements)),
                "improvement_vs_same_nfe_mean": fmt(mean(nfe_improvements)),
                "improvement_vs_same_nfe_std": fmt(sample_std(nfe_improvements)),
                "improvement_vs_same_nfe_min": fmt(min(nfe_improvements)),
                "improvement_vs_same_nfe_max": fmt(max(nfe_improvements)),
                "seed_values": ";".join(
                    f"{seed}:restart={fmt(by_seed[seed]['restart_fid'])},"
                    f"main={fmt(by_seed[seed]['same_main_ode_fid'])},"
                    f"equal_nfe={fmt(by_seed[seed]['same_nfe_ode_fid'])},"
                    f"delta_nfe={fmt(by_seed[seed]['improvement_vs_same_nfe'])}"
                    for seed in ordered_seeds
                ),
                "complete": int(set(ordered_seeds) == expected_set),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    complete = sum(int(row["complete"]) for row in output_rows)
    print(f"Wrote {len(output_rows)} rows ({complete} complete) to {args.output}")


if __name__ == "__main__":
    main()
