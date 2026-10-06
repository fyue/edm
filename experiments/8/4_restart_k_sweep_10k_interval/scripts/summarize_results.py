#!/usr/bin/env python3
"""Summarize Experiment 8-4 K sweeps and select the minimum-FID K."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXP_ROOT / "results" / "restart_fid_summary.tsv"
DEFAULT_SWEEP_OUTPUT = EXP_ROOT / "results" / "restart_k_sweep.tsv"
DEFAULT_OPTIMAL_OUTPUT = EXP_ROOT / "results" / "optimal_k.tsv"
EXPECTED_K = tuple(range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sweep-output", type=Path, default=DEFAULT_SWEEP_OUTPUT)
    parser.add_argument("--optimal-output", type=Path, default=DEFAULT_OPTIMAL_OUTPUT)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--seeds", default="0-4999")
    return parser.parse_args()


def checkpoint_kimg(label: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)k", label)
    if match is None:
        raise ValueError(f"Unsupported checkpoint label: {label!r}")
    return round(float(match.group(1)) * 1000)


def format_number(value: float) -> str:
    return f"{value:.8g}"


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        candidates = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("num_images") == str(args.num_images)
            and row.get("seeds") == args.seeds
        ]

    indexed: dict[tuple[str, int], dict[str, str]] = {}
    for row in candidates:
        try:
            restart_k = int(row["restart_k"])
            fid = float(row["fid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result row: {row}") from exc
        if restart_k not in EXPECTED_K or not math.isfinite(fid):
            raise ValueError(f"Out-of-range K or non-finite FID: {row}")
        key = row["checkpoint"], restart_k
        if key in indexed:
            raise ValueError(f"Duplicate checkpoint/K result: {key}")
        indexed[key] = row

    checkpoints = sorted({key[0] for key in indexed}, key=checkpoint_kimg)
    sweep_rows: list[dict[str, str]] = []
    optimal_rows: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        by_k = {
            restart_k: indexed[(checkpoint, restart_k)]
            for restart_k in EXPECTED_K
            if (checkpoint, restart_k) in indexed
        }
        complete = tuple(sorted(by_k)) == EXPECTED_K
        ode_fid = float(by_k[0]["fid"]) if 0 in by_k else None
        best_k = (
            min(EXPECTED_K, key=lambda restart_k: float(by_k[restart_k]["fid"]))
            if complete
            else None
        )
        for restart_k, row in sorted(by_k.items()):
            fid = float(row["fid"])
            sweep_rows.append(
                {
                    "checkpoint": checkpoint,
                    "training_kimg": str(checkpoint_kimg(checkpoint)),
                    "restart_k": str(restart_k),
                    "nfe": row["nfe"],
                    "fid": format_number(fid),
                    "delta_vs_k0": (
                        "NA" if ode_fid is None else format_number(fid - ode_fid)
                    ),
                    "is_best": "1" if restart_k == best_k else "0",
                    "complete_sweep": "1" if complete else "0",
                    "num_images": row["num_images"],
                    "seeds": row["seeds"],
                    "fid_ref": row["fid_ref"],
                }
            )
        if complete:
            best_row = by_k[best_k]
            best_fid = float(best_row["fid"])
            optimal_rows.append(
                {
                    "checkpoint": checkpoint,
                    "training_kimg": str(checkpoint_kimg(checkpoint)),
                    "optimal_k": str(best_k),
                    "optimal_nfe": best_row["nfe"],
                    "optimal_fid": format_number(best_fid),
                    "k0_fid": format_number(ode_fid),
                    "improvement_vs_k0": format_number(ode_fid - best_fid),
                    "num_images": best_row["num_images"],
                    "seeds": best_row["seeds"],
                    "fid_ref": best_row["fid_ref"],
                }
            )

    sweep_fields = [
        "checkpoint",
        "training_kimg",
        "restart_k",
        "nfe",
        "fid",
        "delta_vs_k0",
        "is_best",
        "complete_sweep",
        "num_images",
        "seeds",
        "fid_ref",
    ]
    optimal_fields = [
        "checkpoint",
        "training_kimg",
        "optimal_k",
        "optimal_nfe",
        "optimal_fid",
        "k0_fid",
        "improvement_vs_k0",
        "num_images",
        "seeds",
        "fid_ref",
    ]
    for path, rows, fields in [
        (args.sweep_output, sweep_rows, sweep_fields),
        (args.optimal_output, optimal_rows, optimal_fields),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
