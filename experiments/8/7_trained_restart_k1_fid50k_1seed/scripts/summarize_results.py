#!/usr/bin/env python3
"""Summarize single-range Restart K=1 improvement against the 39-NFE ODE."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["checkpoint", "num_images", "seeds", "nfe", "fid_ref", "restart_fid",
          "same_nfe_ode_fid", "improvement_vs_same_nfe", "complete"]


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        if row["num_images"] != "50000" or row["seeds"] != "0-49999" or row["nfe"] != "39":
            raise ValueError("Experiment 8-7 requires FID-50k, seeds 0-49999, and 39 NFE")
        method = row["method"]
        if method not in ("restart", "ode-heun"):
            raise ValueError(f"Unexpected method: {method}")
        if not math.isfinite(float(row["fid"])):
            raise ValueError(f"Non-finite FID: {row}")
        expected = ("18", "3", "1") if method == "restart" else ("20", "0", "0")
        if tuple(row[field] for field in ("main_steps", "restart_steps", "restart_k")) != expected:
            raise ValueError(f"Unexpected sampling settings: {row}")
        key = tuple(row[field] for field in ("checkpoint", "num_images", "seeds", "nfe", "fid_ref"))
        pair = grouped.setdefault(key, {})
        if method in pair:
            raise ValueError(f"Duplicate result for {key}, {method}")
        pair[method] = row
    results = []
    for key, pair in sorted(grouped.items()):
        complete = len(pair) == 2
        if complete:
            for field in ("network", "rho", "restart_noise", "gen_batch", "fid_batch"):
                if pair["restart"][field] != pair["ode-heun"][field]:
                    raise ValueError(f"Mismatched {field} in {key}")
        restart = float(pair["restart"]["fid"]) if "restart" in pair else None
        ode = float(pair["ode-heun"]["fid"]) if "ode-heun" in pair else None
        values = dict(zip(FIELDS[:5], key))
        values.update(restart_fid="NA" if restart is None else f"{restart:.8g}",
                      same_nfe_ode_fid="NA" if ode is None else f"{ode:.8g}",
                      improvement_vs_same_nfe=f"{ode - restart:.8g}" if complete else "NA",
                      complete=int(complete))
        results.append(values)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "restart_fid_summary.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "restart_improvements.tsv")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        results = summarize(list(csv.DictReader(handle, delimiter="\t")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} checkpoint comparisons ({sum(row['complete'] for row in results)} complete) to {args.output}")


if __name__ == "__main__":
    main()
