#!/usr/bin/env python3
"""Validate that a generated image pool contains exactly seeds 0..N-1."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path)
    parser.add_argument("--count", type=int, required=True)
    return parser.parse_args()


def preview(values: list[int | str], limit: int = 10) -> str:
    shown = ", ".join(map(str, values[:limit]))
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if not args.pool.is_dir():
        raise SystemExit(f"Missing image pool: {args.pool}")

    seen: dict[int, Path] = {}
    invalid: list[str] = []
    duplicates: list[int] = []
    for path in args.pool.rglob("*.png"):
        try:
            seed = int(path.stem)
        except ValueError:
            invalid.append(str(path))
            continue
        if f"{seed:06d}" != path.stem:
            invalid.append(str(path))
            continue
        if seed in seen:
            duplicates.append(seed)
            continue
        seen[seed] = path

    expected = set(range(args.count))
    actual = set(seen)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    errors = []
    if invalid:
        errors.append(f"invalid filenames: {preview(sorted(invalid))}")
    if duplicates:
        errors.append(f"duplicate seeds: {preview(sorted(set(duplicates)))}")
    if missing:
        errors.append(f"missing seeds: {preview(missing)}")
    if unexpected:
        errors.append(f"unexpected seeds: {preview(unexpected)}")
    if errors:
        raise SystemExit("; ".join(errors))
    print(f"Validated {args.count} images in {args.pool}")


if __name__ == "__main__":
    main()
