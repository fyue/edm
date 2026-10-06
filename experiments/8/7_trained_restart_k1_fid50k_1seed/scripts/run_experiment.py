#!/usr/bin/env python3
"""Experiment 8-7: evaluate trained checkpoints at 39 NFE over one seed range.

Use schedule_after_gpu_free.sh to run GPU work. Environment options:
PLAN_ONLY, SAMPLES_ONLY, SAMPLE_SEEDS, CHECKPOINT_FILTER, CONDITION_FILTER,
EXP_ROOT, CONFIG, FID_REF, GEN_BATCH, FID_BATCH, PYTHON, TORCHRUN, DNNLIB_CACHE_DIR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[3]
DEFAULT_ROOT = SCRIPT_ROOT.parent
NUM_IMAGES = 50000
FID_SEEDS = "0-49999"
FIELDS = [
    "condition", "checkpoint", "method", "main_steps", "restart_steps",
    "restart_k", "t_min", "t_max", "nfe", "comparison_group", "num_images",
    "seeds", "fid", "generate_sec", "fid_sec", "network", "images", "log",
    "fid_ref", "rho", "restart_noise", "gen_batch", "fid_batch", "protocol_id",
]
INTEGER_FIELDS = ["main_steps", "restart_steps", "restart_k", "nfe"]
SCALAR = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?$")


def flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def parse_seeds(value: str) -> list[int]:
    match = re.fullmatch(r"(\d+)-(\d+)", value)
    if match is None:
        raise ValueError(f"Expected one inclusive seed range: {value}")
    start, end = map(int, match.groups())
    if end < start or end >= 2 ** 32:
        raise ValueError(f"Invalid seed range: {value}")
    return list(range(start, end + 1))


def file_identity(path: str) -> dict:
    if urlparse(path).scheme in ("http", "https"):
        return {"source": path}
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Not a file: {path}")
    stat = resolved.stat()
    return {"source": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_conditions(config: Path) -> list[dict]:
    with config.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty config: {config}")
    seen = set()
    for row in rows:
        for field in ("condition", "checkpoint"):
            if not re.fullmatch(r"[A-Za-z0-9._-]+", row[field]) or row[field] in (".", ".."):
                raise ValueError(f"Invalid {field}: {row[field]}")
        if row["condition"] in seen:
            raise ValueError(f"Duplicate condition: {row['condition']}")
        seen.add(row["condition"])
        for field in INTEGER_FIELDS:
            row[field] = int(row[field])
        row["t_min"], row["t_max"] = float(row["t_min"]), float(row["t_max"])
        expected = 2 * row["main_steps"] - 1 + 2 * row["restart_k"] * (row["restart_steps"] - 1)
        if row["nfe"] != expected or row["nfe"] != 39:
            raise ValueError(f"{row['condition']}: expected 39 NFE, calculated {expected}")
        if row["restart_k"] == 1:
            if (row["main_steps"], row["restart_steps"], row["t_min"], row["t_max"]) != (18, 3, 0.06, 0.30):
                raise ValueError(f"{row['condition']}: require Restart [3,1,0.06,0.30], main steps=18")
        elif row["restart_k"] == 0:
            if (row["main_steps"], row["restart_steps"], row["t_min"], row["t_max"]) != (20, 0, 0, 0):
                raise ValueError(f"{row['condition']}: require 20-step ODE without Restart")
        else:
            raise ValueError(f"{row['condition']}: only K=0 or K=1 is supported")
        row["method"] = "restart" if row["restart_k"] else "ode-heun"
        file_identity(row["network"])
    for checkpoint in {row["checkpoint"] for row in rows}:
        pair = [row for row in rows if row["checkpoint"] == checkpoint]
        if len(pair) != 2 or {row["method"] for row in pair} != {"restart", "ode-heun"}:
            raise ValueError(f"{checkpoint}: config must contain one Restart/ODE pair")
        if len({row["network"] for row in pair}) != 1:
            raise ValueError(f"{checkpoint}: pair must use the same checkpoint source")
    return rows


def select_conditions(rows: list[dict]) -> list[dict]:
    for field, env_name in (("checkpoint", "CHECKPOINT_FILTER"), ("condition", "CONDITION_FILTER")):
        wanted = set(os.environ.get(env_name, "").split())
        unknown = wanted - {row[field] for row in rows}
        if unknown:
            raise ValueError(f"Unknown {env_name}: {sorted(unknown)}")
        if wanted:
            rows = [row for row in rows if row[field] in wanted]
    if not rows:
        raise ValueError("No conditions matched the filters")
    return rows


def validate_images(directory: Path, seeds: list[int], subdirs: bool) -> bool:
    expected = {
        f"{seed - seed % 1000:06d}/{seed:06d}.png" if subdirs else f"{seed:06d}.png"
        for seed in seeds
    }
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*.png")}
    extra = actual - expected
    if extra:
        raise ValueError(f"Unexpected images in {directory}: {sorted(extra)[:5]}")
    return actual == expected


def protocol(row: dict, ref: str, gen_batch: int, fid_batch: int, seeds: str) -> dict:
    return {
        "condition": row, "network_identity": file_identity(row["network"]),
        "fid_ref": file_identity(ref), "rho": 7, "restart_noise": 1,
        "gen_batch": gen_batch, "fid_batch": fid_batch, "seeds": seeds,
        "generate_source": file_identity(str(REPO_ROOT / "generate.py")),
        "fid_source": file_identity(str(REPO_ROOT / "fid.py")),
    }


def protocol_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def ensure_protocol(directory: Path, value: dict) -> None:
    metadata = directory / ".protocol.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Protocol changed for {directory}; use a new EXP_ROOT")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Missing protocol metadata for nonempty {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(command: list[str], log: Path) -> tuple[float, str | None]:
    print(shlex.join(command), flush=True)
    started = time.monotonic()
    scalar = None
    with log.open("w", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        with subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
                if SCALAR.fullmatch(line.strip()):
                    scalar = line.strip()
            if process.wait() != 0:
                raise RuntimeError(f"Command failed; see {log}")
    return time.monotonic() - started, scalar


def generate_command(row: dict, directory: Path, seeds: str, batch: int,
                     torchrun: str, subdirs: bool) -> list[str]:
    command = [torchrun, "--standalone", "--nproc_per_node=1", "generate.py",
               f"--network={row['network']}", f"--outdir={directory}", f"--seeds={seeds}",
               f"--batch={batch}", f"--steps={row['main_steps']}", "--rho=7", "--S_churn=0"]
    if row["restart_k"]:
        info = json.dumps({"0": [3, 1, 0.06, 0.30]}, separators=(",", ":"))
        command += ["--sampler=restart", f"--restart_info={info}", "--restart_noise=1"]
    else:
        command += ["--sampler=edm"]
    if subdirs:
        command.append("--subdirs")
    return command


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected summary schema: {path}")
        return list(reader)


def aggregate(root: Path, summary: Path) -> None:
    if not summary.exists():
        return
    output = root / "results" / "restart_improvements.tsv"
    subprocess.run([sys.executable, str(SCRIPT_ROOT / "summarize_results.py"),
                    "--input", str(summary), "--output", str(output)], check=True)
    subprocess.run([sys.executable, str(SCRIPT_ROOT / "plot_results.py"),
                    "--input", str(output), "--output",
                    str(root / "results" / "restart_k1_fid50k_overview")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    os.chdir(REPO_ROOT)
    root = Path(os.environ.get("EXP_ROOT", str(DEFAULT_ROOT))).resolve()
    config = Path(os.environ.get("CONFIG", str(DEFAULT_ROOT / "config" / "restart_k1_fid50k.tsv")))
    ref = os.environ.get("FID_REF", "fid-refs/cifar10-32x32.npz")
    file_identity(ref)
    rows = select_conditions(load_conditions(config))
    plan_only, samples_only = flag("PLAN_ONLY"), flag("SAMPLES_ONLY")
    sample_seeds = os.environ.get("SAMPLE_SEEDS", "0-1" if samples_only else "-")
    samples = parse_seeds(sample_seeds) if sample_seeds != "-" else []
    if samples_only and not samples:
        raise ValueError("SAMPLES_ONLY requires SAMPLE_SEEDS")
    gen_batch, fid_batch = positive_int("GEN_BATCH", 64), positive_int("FID_BATCH", 64)
    torchrun = os.environ.get("TORCHRUN", str(Path(sys.executable).parent / "torchrun"))
    os.environ.setdefault("DNNLIB_CACHE_DIR", str(root / ".cache" / "dnnlib"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    protocols = {row["condition"]: protocol(row, ref, gen_batch, fid_batch, FID_SEEDS) for row in rows}

    if plan_only:
        mode = f"smoke test seeds {sample_seeds}" if samples_only else f"FID-50k seeds {FID_SEEDS}"
        print(f"PLAN Experiment 8-7: {len(rows)} conditions; {mode}; one GPU")
        if not samples_only:
            print(f"PLAN {len(rows)} FID evaluations, {len(rows) * NUM_IMAGES:,} images")
        print(f"PLAN model/detector cache: {os.environ['DNNLIB_CACHE_DIR']}")
        for row in rows:
            print(f"PLAN {row['condition']}: checkpoint={row['checkpoint']} method={row['method']} "
                  f"main_steps={row['main_steps']} K={row['restart_k']} NFE={row['nfe']} network={row['network']}")
            seeds = sample_seeds if samples_only else FID_SEEDS
            print("PLAN " + shlex.join(generate_command(row, root / "images" / row["condition"],
                                                        seeds, gen_batch, torchrun, not samples_only)))
        return

    if shutil.which(torchrun) is None:
        raise ValueError(f"Missing executable torchrun: {torchrun}")
    stop_file = root / "STOP_AFTER_CURRENT_CONDITION"
    if stop_file.exists():
        print(f"Stop file already exists: {stop_file}; no condition started")
        return
    log_dir, results_dir = root / "logs", root / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = results_dir / "restart_fid_summary.tsv"
    recorded = read_results(summary)
    for row in rows:
        condition = row["condition"]
        if samples:
            directory = root / "images" / condition / f"samples-seeds-{sample_seeds}"
            ensure_protocol(directory, protocol(row, ref, gen_batch, fid_batch, sample_seeds))
            if not validate_images(directory, samples, subdirs=False):
                run_logged(generate_command(row, directory, sample_seeds, gen_batch, torchrun, False),
                           log_dir / f"{condition}-samples.log")
            if not validate_images(directory, samples, subdirs=False):
                raise ValueError(f"Incomplete smoke test images: {directory}")
            print(f"[{condition}] fixed-seed samples verified", flush=True)
        if not samples_only:
            value = protocols[condition]
            signature = protocol_id(value)
            matches = [item for item in recorded if item["condition"] == condition]
            if any(item["protocol_id"] != signature for item in matches):
                raise ValueError(f"Recorded protocol changed for {condition}; use a new EXP_ROOT")
            if matches:
                print(f"[{condition}] FID-50k already recorded; skipping", flush=True)
            else:
                directory = root / "images" / condition / f"fid-n50000-seeds-{FID_SEEDS}"
                ensure_protocol(directory, value)
                seeds = parse_seeds(FID_SEEDS)
                generation_sec = "NA"
                if not validate_images(directory, seeds, subdirs=True):
                    seconds, _ = run_logged(generate_command(row, directory, FID_SEEDS, gen_batch, torchrun, True),
                                            log_dir / f"{condition}-generate.log")
                    generation_sec = f"{seconds:.3f}"
                if not validate_images(directory, seeds, subdirs=True):
                    raise ValueError(f"Require exactly {NUM_IMAGES} expected seed images in {directory}")
                fid_log = log_dir / f"{condition}-fid.log"
                seconds, fid = run_logged([torchrun, "--standalone", "--nproc_per_node=1", "fid.py", "calc",
                                           f"--images={directory}", f"--ref={ref}", "--num=50000",
                                           f"--batch={fid_batch}"], fid_log)
                if fid is None or not math.isfinite(float(fid)):
                    raise ValueError(f"No finite FID in {fid_log}")
                result = dict(row, num_images=NUM_IMAGES, seeds=FID_SEEDS, fid=fid,
                              generate_sec=generation_sec, fid_sec=f"{seconds:.3f}",
                              images=str(directory), log=str(fid_log), fid_ref=ref,
                              rho=7, restart_noise=1, gen_batch=gen_batch,
                              fid_batch=fid_batch, protocol_id=signature)
                fresh = not summary.exists()
                with summary.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
                    if fresh:
                        writer.writeheader()
                    writer.writerow(result)
                recorded.append({key: str(result[key]) for key in FIELDS})
                print(f"[{condition}] recorded FID-50k={fid}", flush=True)
                aggregate(root, summary)
        if stop_file.exists():
            print(f"Found {stop_file}; stopping after {condition}", flush=True)
            break
    if not samples_only:
        aggregate(root, summary)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
