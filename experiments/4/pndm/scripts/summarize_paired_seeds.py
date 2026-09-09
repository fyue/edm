#!/usr/bin/env python3
"""Summarize PNDM FID across seed ranges and paired churn improvements."""

import argparse
import csv
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--condition-out", required=True, type=Path)
    parser.add_argument("--improvement-out", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    rows = []
    for path in args.summary:
        rows.extend(read_tsv(path))

    observations = {}
    labels = {}
    for row in rows:
        if row["sampler"] != "pndm":
            continue
        key = (row["checkpoint"], float(row["S_churn"]))
        labels[row["checkpoint"]] = row["label"]
        observations.setdefault(key, {})[row["seeds"]] = float(row["fid"])

    condition_rows = []
    for (checkpoint, churn), by_seed in sorted(observations.items()):
        values = list(by_seed.values())
        condition_rows.append(
            {
                "checkpoint": checkpoint,
                "label": labels[checkpoint],
                "S_churn": f"{churn:g}",
                "num_seed_ranges": len(values),
                "fid_mean": f"{statistics.mean(values):.8g}",
                "fid_std": f"{sample_std(values):.8g}",
                "seed_values": ",".join(
                    f"{seed}:{value:.8g}" for seed, value in sorted(by_seed.items())
                ),
            }
        )

    improvement_rows = []
    for pair in read_tsv(args.pairs):
        if pair["checkpoint"].startswith("#"):
            continue
        checkpoint = pair["checkpoint"]
        baseline = float(pair["baseline_churn"])
        selected = float(pair["selected_churn"])
        baseline_values = observations.get((checkpoint, baseline), {})
        selected_values = observations.get((checkpoint, selected), {})
        common_seeds = sorted(set(baseline_values) & set(selected_values))
        improvements = [
            baseline_values[seed] - selected_values[seed] for seed in common_seeds
        ]
        if not improvements:
            continue
        improvement_rows.append(
            {
                "checkpoint": checkpoint,
                "label": pair["label"],
                "baseline_churn": f"{baseline:g}",
                "selected_churn": f"{selected:g}",
                "num_seed_ranges": len(improvements),
                "improvement_mean": f"{statistics.mean(improvements):.8g}",
                "improvement_std": f"{sample_std(improvements):.8g}",
                "paired_values": ",".join(
                    f"{seed}:{value:.8g}"
                    for seed, value in zip(common_seeds, improvements)
                ),
            }
        )

    write_tsv(
        args.condition_out,
        [
            "checkpoint",
            "label",
            "S_churn",
            "num_seed_ranges",
            "fid_mean",
            "fid_std",
            "seed_values",
        ],
        condition_rows,
    )
    write_tsv(
        args.improvement_out,
        [
            "checkpoint",
            "label",
            "baseline_churn",
            "selected_churn",
            "num_seed_ranges",
            "improvement_mean",
            "improvement_std",
            "paired_values",
        ],
        improvement_rows,
    )


if __name__ == "__main__":
    main()
