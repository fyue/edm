#!/usr/bin/env python3
"""Build K=0..8 conditions from available checkpoints without editing TSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from prepare_fid25k_reuse import CONFIG_FIELDS, load_config


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = EXP_ROOT / "config" / "restart_k0_8_fid5k.tsv"


def parse_selection(selection: str) -> set[int] | None:
    """Accept 160k, lists, inclusive 10k-160k:10k ranges, or all."""
    if selection.strip() == "all":
        return None
    checkpoints = set()
    for token in selection.replace(",", " ").split():
        match = re.fullmatch(r"([1-9][0-9]*)k(?:-([1-9][0-9]*)k(?::([1-9][0-9]*)k)?)?", token)
        if match is None:
            raise ValueError(
                f"Invalid checkpoint selection {token!r}; use '160k', "
                "'120k 160k', '10k-160k:10k', or 'all'"
            )
        start = int(match[1]) * 1000
        end = int(match[2]) * 1000 if match[2] else start
        step = int(match[3]) * 1000 if match[3] else 10000
        if end < start or (end - start) % step:
            raise ValueError(f"Invalid range or unaligned endpoint: {token}")
        checkpoints.update(range(start, end + 1, step))
    if not checkpoints:
        raise ValueError("Checkpoint selection is empty")
    return checkpoints


def completed_checkpoints(summary: Path) -> set[str]:
    if not summary.is_file():
        return set()
    by_checkpoint = {}
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("num_images") != "5000" or row.get("seeds") != "0-4999":
                continue
            by_checkpoint.setdefault(row["checkpoint"], set()).add(int(row["restart_k"]))
    return {checkpoint for checkpoint, ks in by_checkpoint.items() if ks == set(range(9))}


def build_rows(
    template: Path,
    checkpoint_dir: Path | None,
    selection: str,
    source_summary: Path | None,
) -> list[dict[str, str]]:
    template_rows = load_config(template)
    first_checkpoint = template_rows[0]["checkpoint"]
    prototype = [row for row in template_rows if row["checkpoint"] == first_checkpoint]
    if sorted(int(row["restart_k"]) for row in prototype) != list(range(9)):
        raise ValueError(f"Template must contain exactly one row per K=0..8 for {first_checkpoint}")
    checkpoint_dir = checkpoint_dir or Path(prototype[0]["network"]).parent
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    available = {}
    for path in checkpoint_dir.glob("network-snapshot-*.pkl"):
        match = re.fullmatch(r"network-snapshot-([0-9]+)\.pkl", path.name)
        if match and path.is_file():
            step = int(match[1])
            if step in available:
                raise ValueError(f"Duplicate checkpoint step {step}: {available[step]}, {path}")
            available[step] = path

    requested = parse_selection(selection)
    if requested is None:
        requested = {step for step in available if step > 0 and step % 10000 == 0}
        if source_summary is not None:
            complete = completed_checkpoints(source_summary)
            requested = {step for step in requested if f"{step // 1000}k" in complete}
    missing = sorted(requested - available.keys())
    if missing:
        raise ValueError(
            f"Missing requested checkpoints in {checkpoint_dir}: "
            + " ".join(f"{step // 1000}k" for step in missing)
        )
    if not requested:
        raise ValueError(
            "No checkpoints selected. FID-25k automatic selection requires complete "
            "FID-5k K=0..8 results; run the FID-5k runner first."
            if source_summary is not None else "No 10k-interval checkpoints found"
        )

    output = []
    for step in sorted(requested):
        checkpoint = f"{step // 1000}k"
        for row in sorted(prototype, key=lambda row: int(row["restart_k"])):
            k = int(row["restart_k"])
            output.append({
                **row,
                "condition": f"{checkpoint}-ode-k0" if k == 0 else f"{checkpoint}-restart-k{k}",
                "checkpoint": checkpoint,
                "network": str(available[step]),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoints", default="all")
    parser.add_argument("--source-summary", type=Path,
                        help="For 'all', restrict to complete FID-5k checkpoint sweeps")
    args = parser.parse_args()
    rows = build_rows(args.template, args.checkpoint_dir, args.checkpoints, args.source_summary)
    print("# Automatically selected Experiment 8-4 checkpoints; K=0..8.")
    print("#" + "\t".join(CONFIG_FIELDS))
    writer = csv.DictWriter(sys.stdout, fieldnames=CONFIG_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    labels = list(dict.fromkeys(row["checkpoint"] for row in rows))
    print(f"Selected {len(labels)} checkpoints / {len(rows)} conditions: {' '.join(labels)}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
