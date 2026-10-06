#!/usr/bin/env python3
"""Validate and reuse Experiment 8-4 FID-5k images for FID-25k."""

from __future__ import annotations

import argparse
import csv
import errno
import filecmp
import os
import shutil
import tempfile
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXP_ROOT / "config" / "restart_k0_8_fid25k_10k_100k.tsv"
DEFAULT_SOURCE_SUMMARY = EXP_ROOT / "results" / "restart_fid_summary.tsv"
DEFAULT_IMAGE_ROOT = EXP_ROOT / "images"
DEFAULT_MANIFEST = EXP_ROOT / "results" / "fid25k" / "reuse_manifest.tsv"
SOURCE_DIRNAME = "fid-n5000-seeds-0-4999"
DESTINATION_DIRNAME = "fid-n25000-seeds-0-24999"
SOURCE_COUNT = 5000
TARGET_COUNT = 25000
GENERATION_SEEDS = "5000-24999"
CONFIG_FIELDS = [
    "condition",
    "checkpoint",
    "main_steps",
    "restart_steps",
    "restart_k",
    "t_min",
    "t_max",
    "nfe",
    "comparison_group",
    "network",
]
COMPARISON_FIELDS = [
    "checkpoint",
    "main_steps",
    "restart_steps",
    "restart_k",
    "t_min",
    "t_max",
    "nfe",
    "network",
]
MANIFEST_FIELDS = [
    "condition",
    "checkpoint",
    "source_dir",
    "destination_dir",
    "reuse_mode",
    "reused_images",
    "created_files",
    "existing_files",
    "generation_seeds",
    "target_images",
    "source_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-filter", default="")
    parser.add_argument("--condition-filter", default="")
    parser.add_argument("--mode", choices=["check", "materialize"], default="check")
    parser.add_argument("--reuse-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def filter_values(value: str) -> set[str]:
    return set(value.split())


def selected(value: str, wanted: set[str]) -> bool:
    return not wanted or value in wanted


def load_config(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            values = line.split("\t")
            if len(values) != len(CONFIG_FIELDS):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(CONFIG_FIELDS)} fields, "
                    f"got {len(values)}"
                )
            rows.append(dict(zip(CONFIG_FIELDS, values)))
    if not rows:
        raise ValueError(f"No conditions in {path}")
    return rows


def load_source_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("num_images") == str(SOURCE_COUNT)
            and row.get("seeds") == "0-4999"
        ]
    indexed = {}
    for row in rows:
        condition = row["condition"]
        if condition in indexed:
            raise ValueError(f"Duplicate FID-5k result for {condition} in {path}")
        indexed[condition] = row
    return indexed


def expected_relative_path(seed: int) -> Path:
    subdir = seed - seed % 1000
    return Path(f"{subdir:06d}") / f"{seed:06d}.png"


def validate_source_images(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise ValueError(f"Missing FID-5k source directory: {source_dir}")
    pngs = sorted(source_dir.rglob("*.png"))
    if len(pngs) != SOURCE_COUNT:
        raise ValueError(
            f"Expected {SOURCE_COUNT} source PNGs in {source_dir}, got {len(pngs)}"
        )
    actual = {path.relative_to(source_dir) for path in pngs}
    expected = {expected_relative_path(seed) for seed in range(SOURCE_COUNT)}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"Invalid seed coverage in {source_dir}: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return [source_dir / expected_relative_path(seed) for seed in range(SOURCE_COUNT)]


def validate_source_result(
    config_row: dict[str, str],
    summary_rows: dict[str, dict[str, str]],
    summary_path: Path,
) -> None:
    condition = config_row["condition"]
    result_row = summary_rows.get(condition)
    if result_row is None:
        raise ValueError(
            f"Missing FID-5k result for {condition} in {summary_path}. "
            f"First run scripts/schedule_after_gpu_free.sh with "
            f"CHECKPOINT_FILTER=\"{config_row['checkpoint']}\" to prepare FID-5k images/results."
        )
    mismatches = [
        field
        for field in COMPARISON_FIELDS
        if result_row.get(field) != config_row[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: config={config_row[field]!r}, result={result_row.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"FID-5k provenance mismatch for {condition}: {details}")


def same_contents(source: Path, destination: Path) -> bool:
    try:
        if os.path.samefile(source, destination):
            return True
    except FileNotFoundError:
        return False
    return filecmp.cmp(source, destination, shallow=False)


def reuse_images(
    source_paths: list[Path],
    source_dir: Path,
    destination_dir: Path,
    reuse_mode: str,
) -> tuple[int, int]:
    created = 0
    existing = 0
    for source in source_paths:
        destination = destination_dir / source.relative_to(source_dir)
        if destination.exists():
            if not same_contents(source, destination):
                raise ValueError(
                    f"Destination collision with different contents: {destination}"
                )
            existing += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if reuse_mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise ValueError(
                        "Cannot create hard links across filesystems; "
                        "rerun explicitly with --reuse-mode=copy"
                    ) from exc
                raise
        else:
            shutil.copy2(source, destination)
        created += 1
    return created, existing


def validate_destination_reuse(source_dir: Path, destination_dir: Path) -> None:
    for seed in range(SOURCE_COUNT):
        relative = expected_relative_path(seed)
        source = source_dir / relative
        destination = destination_dir / relative
        if not destination.is_file():
            raise ValueError(f"Missing reused destination image: {destination}")
        if not same_contents(source, destination):
            raise ValueError(f"Reused image differs from source: {destination}")


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["condition"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def write_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                rows.values(),
                key=lambda row: (
                    int(row["checkpoint"].removesuffix("k")),
                    row["condition"],
                ),
            )
        )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    checkpoint_filter = filter_values(args.checkpoint_filter)
    condition_filter = filter_values(args.condition_filter)
    config_rows = [
        row
        for row in load_config(args.config)
        if selected(row["checkpoint"], checkpoint_filter)
        and selected(row["condition"], condition_filter)
    ]
    if not config_rows:
        raise ValueError("No FID-25k conditions matched the requested filters")

    summary_rows = load_source_summary(args.source_summary)
    manifest_rows = load_manifest(args.manifest) if args.mode == "materialize" else {}
    for row in config_rows:
        condition = row["condition"]
        validate_source_result(row, summary_rows, args.source_summary)
        source_dir = args.image_root / condition / SOURCE_DIRNAME
        destination_dir = args.image_root / condition / DESTINATION_DIRNAME
        source_paths = validate_source_images(source_dir)
        if args.mode == "check":
            print(
                f"CHECK [{condition} checkpoint={row['checkpoint']}] "
                f"reuse={SOURCE_COUNT} generate={GENERATION_SEEDS} target={TARGET_COUNT}"
            )
            continue

        if args.reuse_mode == "hardlink":
            if source_dir.stat().st_dev != args.image_root.stat().st_dev:
                raise ValueError(
                    f"Hard links cross filesystems for {condition}; rerun with --reuse-mode=copy"
                )
        created, existing = reuse_images(
            source_paths,
            source_dir,
            destination_dir,
            args.reuse_mode,
        )
        validate_destination_reuse(source_dir, destination_dir)
        manifest_rows[condition] = {
            "condition": condition,
            "checkpoint": row["checkpoint"],
            "source_dir": str(source_dir),
            "destination_dir": str(destination_dir),
            "reuse_mode": args.reuse_mode,
            "reused_images": str(SOURCE_COUNT),
            "created_files": str(created),
            "existing_files": str(existing),
            "generation_seeds": GENERATION_SEEDS,
            "target_images": str(TARGET_COUNT),
            "source_summary": str(args.source_summary),
        }
        print(
            f"PREPARED [{condition}] created={created} existing={existing} "
            f"destination={destination_dir}"
        )

    if args.mode == "materialize":
        write_manifest(args.manifest, manifest_rows)
        print(f"Wrote {len(manifest_rows)} rows to {args.manifest}")
    else:
        print(f"Validated {len(config_rows)} FID-5k source conditions")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
