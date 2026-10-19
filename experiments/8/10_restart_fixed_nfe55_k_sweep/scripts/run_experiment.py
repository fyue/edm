#!/usr/bin/env python3
"""Experiment 8-10: fixed NFE=55, K sweep excluding K=5.

GPU work must enter through run.sh and the shared GPU waiter.
"""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
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


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[3]
DEFAULT_ROOT = SCRIPT_ROOT.parent
MAX_IMAGES = 25000
STAGES = (5000, 25000)
CHECKPOINTS = tuple(f"{step}k" for step in range(20, 201, 20))
KS = (0, 1, 2, 3, 4, 6, 7, 8)

FIELDS = [
    "condition", "checkpoint", "main_steps", "restart_steps", "restart_k",
    "t_min", "t_max", "nfe", "actual_t_min", "restart_index", "num_images", "seeds", "fid", "generate_sec",
    "fid_sec", "reused_images", "reuse_source", "network", "images", "log",
    "fid_ref", "gen_batch", "fid_batch", "rho", "restart_noise", "protocol_id",
]


def env_flag(name, default=False):
    value = os.environ.get(name, "1" if default else "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def read_tsv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def schedule_point(main_steps):
    # EDMPrecond.round_sigma is identity for these checkpoints.
    times = [(80 ** (1 / 7) + i / (main_steps - 1)
              * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
             for i in range(main_steps)] + [0.0]
    index = min(range(len(times)), key=lambda i: abs(times[i] - 0.06))
    if not 0 < index < main_steps or not 0 < times[index] < 0.30:
        raise ValueError("Invalid Restart schedule point")
    return times[index], index


def load_conditions(path):
    rows = read_tsv(path)
    expected = {(checkpoint, k) for checkpoint in CHECKPOINTS for k in KS}
    seen, names = set(), set()
    for row in rows:
        for field in ("main_steps", "restart_steps", "restart_k", "nfe"):
            row[field] = int(row[field])
        for field in ("t_min", "t_max"):
            row[field] = float(row[field])
        k = row["restart_k"]
        pair = row["checkpoint"], k
        if pair not in expected or pair in seen or row["condition"] in names:
            raise ValueError("Require 20k..200k every 20k x K=0,1,2,3,4,6,7,8")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", row["condition"]):
            raise ValueError("Unsafe condition name")
        seen.add(pair)
        names.add(row["condition"])
        target = (28 - 2 * k, 3 if k else 0, 0.06 if k else 0, 0.30 if k else 0, 55)
        if tuple(row[field] for field in ("main_steps", "restart_steps", "t_min", "t_max", "nfe")) != target:
            raise ValueError("Invalid fixed NFE=55 sampler configuration")
        if 2 * row["main_steps"] - 1 + 4 * k != 55:
            raise ValueError("NFE mismatch")
        row["actual_t_min"], row["restart_index"] = schedule_point(row["main_steps"]) if k else (None, None)
        row["network"] = str((REPO_ROOT / row["network"]).resolve())
        if Path(row["network"]).name != f"network-snapshot-{int(row['checkpoint'][:-1]) * 1000:06d}.pkl":
            raise ValueError("Checkpoint label and snapshot disagree")
        identity(row["network"])
    if seen != expected:
        raise ValueError("Incomplete 80-condition configuration")
    for checkpoint in CHECKPOINTS:
        if len({row["network"] for row in rows if row["checkpoint"] == checkpoint}) != 1:
            raise ValueError("All K values must use the same checkpoint")
    return rows


def select_conditions(rows, checkpoints, conditions):
    for field, value in (("checkpoint", checkpoints), ("condition", conditions)):
        wanted = set(value.replace(",", " ").split())
        unknown = wanted - {row[field] for row in rows}
        if unknown:
            raise ValueError(f"Unknown {field}: {sorted(unknown)}")
        if wanted:
            rows = [row for row in rows if row[field] in wanted]
    if not rows:
        raise ValueError("No conditions matched")
    return rows


def parse_seeds(value):
    match = re.fullmatch(r"(\d+)-(\d+)", value)
    if match is None:
        raise ValueError("Sample seeds must be one inclusive range, e.g. 0-1")
    start, end = map(int, match.groups())
    if start > end or end >= 2 ** 32:
        raise ValueError("Invalid sample seed range")
    return list(range(start, end + 1))


def image_relative(seed):
    return Path(f"{seed - seed % 1000:06d}") / f"{seed:06d}.png"


def validate_png(path):
    from PIL import Image
    if path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"Invalid image file: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (32, 32) or image.mode != "RGB":
            raise ValueError(f"Expected RGB CIFAR-10 PNG: {path}")
        image.verify()


def inventory(directory, allowed, complete=False, verify=True):
    allowed = set(allowed)
    actual = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix != ".png" or not re.fullmatch(r"\d{6,10}", path.stem):
            raise ValueError(f"Unexpected image-pool file: {path}")
        seed = int(path.stem)
        if seed not in allowed or path.relative_to(directory) != image_relative(seed):
            raise ValueError(f"Unexpected seed/path: {path}")
        if path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"Invalid image file: {path}")
        if verify:
            validate_png(path)
        actual[seed] = path
    if complete and set(actual) != allowed:
        raise ValueError(f"{directory}: {len(actual)}/{len(allowed)} images; missing seeds {sorted(allowed - set(actual))[:5]}")
    return actual


def link_image(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target) and source.read_bytes() != target.read_bytes():
            raise ValueError(f"Existing linked image differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM):
            raise
        shutil.copy2(source, target)


def protocol(row, args, seeds=None):
    value = {"condition": row, "network": identity(row["network"]),
            "fid_ref": identity(args.fid_ref), "rho": 7, "restart_noise": 1,
            "sigma_min": 0.002, "sigma_max": 80, "S_churn": 0,
            "gen_batch": args.gen_batch, "fid_batch": args.fid_batch,
            "seeds": seeds or "0-24999", "generate": identity(REPO_ROOT / "generate.py"),
            "fid": identity(REPO_ROOT / "fid.py"), "runner": identity(Path(__file__))}
    directory = "pool" if seeds is None else f"samples-seeds-{seeds}"
    path = args.exp_root / "images" / row["condition"] / directory / ".protocol.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        relocated = json.loads(json.dumps(recorded))
        # Preserve legacy IDs while comparing repo-relative paths across synced PCs.
        old_root = Path(recorded["generate"]["path"]).parent
        try:
            for key in ("network", "fid_ref", "generate", "fid", "runner"):
                relative = Path(recorded[key]["path"]).relative_to(old_root)
                relocated[key]["path"] = str(REPO_ROOT / relative)
            relative = Path(recorded["condition"]["network"]).relative_to(old_root)
            relocated["condition"]["network"] = str(REPO_ROOT / relative)
        except ValueError:
            return value
        if relocated == value:
            return recorded
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ensure_protocol(directory, value, create=True):
    path = directory / ".protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Protocol mismatch in {directory}; choose a new EXP_ROOT")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Missing protocol metadata in {directory}")
    elif create:
        write_json(path, value)


def generate_command(row, directory, seeds, args):
    command = [args.torchrun, "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
               "--rdzv_id=experiment8-10", "--nproc_per_node=1", "generate.py",
               f"--network={row['network']}", f"--outdir={directory}", f"--seeds={seeds}",
               f"--batch={args.gen_batch}", f"--steps={row['main_steps']}", "--rho=7",
               "--sigma_min=0.002", "--sigma_max=80", "--S_churn=0", "--subdirs"]
    if row["restart_k"]:
        info = json.dumps({"0": [row["restart_steps"], row["restart_k"], row["t_min"], row["t_max"]]}, separators=(",", ":"))
        command.extend(["--sampler=restart", f"--restart_info={info}", "--restart_noise=1"])
    else:
        command.append("--sampler=edm")
    return command


def seed_ranges(seeds):
    """Generate only missing seeds without overwriting reused hardlinks."""
    if not seeds:
        return []
    start = previous = seeds[0]
    result = []
    for seed in seeds[1:]:
        if seed != previous + 1:
            result.append(f"{start}-{previous}")
            start = seed
        previous = seed
    result.append(f"{start}-{previous}")
    return result


def run_logged(command, log):
    print(shlex.join(command), flush=True)
    started, scalar = time.monotonic(), None
    with log.open("a", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        with subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
                if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line.strip()):
                    scalar = float(line.strip())
            if process.wait():
                raise RuntimeError(f"Command failed; see {log}")
    return time.monotonic() - started, scalar


def comparison_rows(results):
    groups = {}
    for row in results:
        key = row["checkpoint"], int(row["num_images"])
        k = int(row["restart_k"])
        if row["checkpoint"] not in CHECKPOINTS or k not in KS:
            raise ValueError("Unexpected result checkpoint/K")
        group = groups.setdefault(key, {})
        if k in group:
            raise ValueError("Duplicate completed result")
        group[k] = row
    return groups


def aggregate(root, results):
    groups = comparison_rows(results)
    destination = root / "results"
    results = sorted(results, key=lambda row: (int(row["num_images"]), int(row["checkpoint"][:-1]), int(row["restart_k"])))
    write_tsv(destination / "restart_fid_summary.tsv", results, FIELDS)
    write_json(destination / "restart_fid_summary.json", results)
    table, optimal = [], []
    for (checkpoint, count), group in sorted(groups.items(), key=lambda item: (item[0][1], int(item[0][0][:-1]))):
        baseline = float(group[0]["fid"]) if 0 in group else None
        for k, row in sorted(group.items()):
            table.append(dict(checkpoint=checkpoint, num_images=count, restart_k=k,
                              fid=row["fid"], delta_vs_k0=float(row["fid"]) - baseline if baseline is not None else "NA"))
        if set(group) == set(KS):
            best = min(KS, key=lambda k: (float(group[k]["fid"]), k))
            optimal.append(dict(checkpoint=checkpoint, num_images=count, optimal_k=best,
                                fid=group[best]["fid"], candidate_ks=",".join(map(str, KS))))
    write_tsv(destination / "fid_by_checkpoint_and_k.tsv", table,
              ["checkpoint", "num_images", "restart_k", "fid", "delta_vs_k0"])
    write_tsv(destination / "optimal_k.tsv", optimal,
              ["checkpoint", "num_images", "optimal_k", "fid", "candidate_ks"])
    if not results:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(root / "logs" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for count in sorted({int(row["num_images"]) for row in results}):
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        matrix = np.full((len(CHECKPOINTS), len(KS)), np.nan)
        for i, checkpoint in enumerate(CHECKPOINTS):
            group = groups.get((checkpoint, count), {})
            if not group:
                continue
            xs = sorted(group)
            axes[0].plot(xs, [group[k]["fid"] for k in xs], marker="o", label=checkpoint)
            if 0 in group:
                for j, k in enumerate(KS):
                    if k in group:
                        matrix[i, j] = float(group[k]["fid"]) - float(group[0]["fid"])
        axes[0].set(xlabel="K", ylabel="FID", title=f"FID-{count}, NFE=55")
        axes[0].set_xticks(KS)
        axes[0].legend(fontsize=8)
        heatmap = axes[1].imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="coolwarm")
        axes[1].set_xticks(range(len(KS)), KS)
        axes[1].set_yticks(range(len(CHECKPOINTS)), CHECKPOINTS)
        axes[1].set(xlabel="K", ylabel="Checkpoint (kimg)", title="FID(K) - FID(0)")
        figure.colorbar(heatmap, ax=axes[1])
        complete = [row for row in optimal if row["num_images"] == count]
        axes[2].plot([int(row["checkpoint"][:-1]) for row in complete],
                     [row["optimal_k"] for row in complete], marker="o")
        axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among 8 candidates (K=5 excluded)")
        axes[2].set_yticks(KS)
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(destination / f"fixed_nfe55_fid{count}.{extension}", dpi=180)
        plt.close(figure)


def evaluate(row, count, args, results):
    value = protocol(row, args)
    pool = args.exp_root / "images" / row["condition"] / "pool"
    ensure_protocol(pool, value)
    run_id = digest(value)
    recorded = [item for item in results if item["condition"] == row["condition"] and int(item["num_images"]) == count]
    if len(recorded) > 1:
        raise ValueError("Duplicate summary result")
    view = pool.parent / f"fid-n{count}-seeds-0-{count - 1}"
    existing = inventory(pool, range(MAX_IMAGES))
    previous = {5000: 0, 25000: 5000}[count]
    if not set(range(previous)).issubset(existing):
        raise ValueError(f"{count} extension requires all first {previous} images")
    if recorded:
        if recorded[0]["protocol_id"] != run_id:
            raise ValueError("Recorded result protocol changed")
        ensure_protocol(view, {"pool_protocol_id": run_id, "num_images": count}, create=False)
        inventory(view, range(count), complete=True)
        inventory(pool, range(MAX_IMAGES))
        if not set(range(count)).issubset(existing):
            raise ValueError("Recorded result has missing pool images")
        for seed in range(count):
            link_image(existing[seed], view / image_relative(seed))
        print(f"RESUME {row['condition']} FID-{count}: verified; skipping", flush=True)
        return
    log = args.exp_root / "logs" / f"{row['condition']}-fid{count}.log"
    missing = sorted(set(range(count)) - set(existing))
    generation_sec = 0.0
    for seeds in seed_ranges(missing):
        elapsed, _ = run_logged(generate_command(row, pool, seeds, args), log)
        generation_sec += elapsed
    existing = inventory(pool, range(MAX_IMAGES))
    if not set(range(count)).issubset(existing):
        raise ValueError("Generation left missing seeds")
    ensure_protocol(view, {"pool_protocol_id": run_id, "num_images": count})
    for seed in range(count):
        link_image(existing[seed], view / image_relative(seed))
    inventory(view, range(count), complete=True)
    command = [args.torchrun, "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
               "--rdzv_id=experiment8-10", "--nproc_per_node=1", "fid.py", "calc",
               f"--images={view}", f"--ref={args.fid_ref}", f"--num={count}", f"--batch={args.fid_batch}"]
    fid_sec, fid = run_logged(command, log)
    if fid is None or not math.isfinite(fid):
        raise ValueError(f"Could not parse finite FID from {log}")
    result = dict(row, num_images=count, seeds=f"0-{count - 1}", fid=fid,
                  generate_sec=generation_sec, fid_sec=fid_sec,
                  reused_images=0, reuse_source="",
                  images=str(view), log=str(log), fid_ref=str(args.fid_ref),
                  gen_batch=args.gen_batch, fid_batch=args.fid_batch, rho=7,
                  restart_noise=1, protocol_id=run_id)
    results.append(result)
    # Per-result JSON is the completion marker; interrupted TSV/plot updates recover from it.
    write_json(args.exp_root / "results" / f"{row['condition']}-fid{count}.json", result)
    aggregate(args.exp_root, results)


def load_results(root):
    results = []
    for path in sorted((root / "results").glob("*-fid[0-9]*.json")):
        row = json.loads(path.read_text())
        if set(row) != set(FIELDS) or int(row["num_images"]) not in STAGES or not math.isfinite(float(row["fid"])):
            raise ValueError(f"Invalid completed result: {path}")
        results.append(row)
    comparison_rows(results)
    return results


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fid-stage", choices=("5000", "25000", "all"), default=os.environ.get("FID_STAGE", "5000"),
                        help="run 5k first; 25000/all extends pools to 25k")
    parser.add_argument("--checkpoint-filter", default=os.environ.get("CHECKPOINT_FILTER", ""))
    parser.add_argument("--condition-filter", default=os.environ.get("CONDITION_FILTER", ""))
    parser.add_argument("--exp-root", type=Path, default=Path(os.environ.get("EXP_ROOT", str(DEFAULT_ROOT))))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("CONFIG", str(DEFAULT_ROOT / "config/restart_fixed_nfe55.tsv"))))
    parser.add_argument("--fid-ref", type=Path, default=Path(os.environ.get("FID_REF", "fid-refs/cifar10-32x32.npz")))
    parser.add_argument("--gen-batch", type=int, default=int(os.environ.get("GEN_BATCH", "64")))
    parser.add_argument("--fid-batch", type=int, default=int(os.environ.get("FID_BATCH", "64")))
    parser.add_argument("--torchrun", default=os.environ.get("TORCHRUN", str(Path(sys.executable).parent / "torchrun")))
    parser.add_argument("--resume", dest="resume", action="store_true", default=env_flag("RESUME", True))
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="refuse existing run data; never overwrite")
    parser.add_argument("--sample-seeds", default=os.environ.get("SAMPLE_SEEDS", "0-1"))
    modes = parser.add_mutually_exclusive_group()
    for option, env in (("plan-only", "PLAN_ONLY"), ("verify-only", "VERIFY_ONLY"), ("aggregate-only", "AGGREGATE_ONLY")):
        modes.add_argument(f"--{option}", action="store_true", default=env_flag(env))
    parser.add_argument("--samples-only", action="store_true", default=env_flag("SAMPLES_ONLY"))
    args = parser.parse_args()
    if sum((args.plan_only, args.verify_only, args.aggregate_only)) > 1:
        parser.error("plan-only, verify-only and aggregate-only are mutually exclusive")
    if args.gen_batch < 1 or args.fid_batch < 1:
        parser.error("batch sizes must be positive")
    if args.fid_stage not in ("5000", "25000", "all"):
        parser.error("invalid FID_STAGE")
    return args


def main():
    os.chdir(REPO_ROOT)
    args = arguments()
    args.exp_root = args.exp_root.resolve()
    args.fid_ref = args.fid_ref.resolve()
    rows = select_conditions(load_conditions(args.config), args.checkpoint_filter, args.condition_filter)
    identity(args.fid_ref)
    results = load_results(args.exp_root)
    if args.aggregate_only:
        aggregate(args.exp_root, results)
        print(f"Aggregated {len(results)} completed evaluations")
        return
    maximum = 25000 if args.fid_stage == "all" else int(args.fid_stage)
    stages = [stage for stage in STAGES if stage <= maximum]
    sample_seeds = parse_seeds(args.sample_seeds)
    if args.plan_only or args.verify_only:
        total_new = 0
        pending_evaluations = 0
        for row in rows:
            value = protocol(row, args)
            directory = args.exp_root / "images" / row["condition"] / "pool"
            ensure_protocol(directory, value, create=False)
            images = inventory(directory, range(MAX_IMAGES))
            print(f"PLAN {row['condition']}: checkpoint={row['checkpoint']} t_min={row['t_min']} t_max={row['t_max']} "
                  f"main={row['main_steps']} restart_steps={row['restart_steps']} K={row['restart_k']} "
                  f"NFE=55 actual_t_min={row['actual_t_min']} restart_index={row['restart_index']} "
                  f"pool={directory} existing={len(images)}")
            if args.samples_only:
                sample_dir = args.exp_root / "images" / row["condition"] / f"samples-seeds-{args.sample_seeds}"
                ensure_protocol(sample_dir, protocol(row, args, args.sample_seeds), create=False)
                sample_images = inventory(sample_dir, sample_seeds, complete=args.verify_only)
                print(f"SAMPLES existing={len(sample_images)}/{len(sample_seeds)}; directory={sample_dir}")
                print("PLAN " + shlex.join(generate_command(row, sample_dir, args.sample_seeds, args)))
            else:
                available = set(images)
                for count in stages:
                    missing = sorted(set(range(count)) - available)
                    total_new += len(missing)
                    done = [item for item in results if item["condition"] == row["condition"] and int(item["num_images"]) == count]
                    pending_evaluations += not bool(done)
                    print(f"PLAN FID-{count}: seeds 0-{count - 1}; new_images={len(missing)}; "
                          f"evaluation={'resume/verify' if done else 'pending'}")
                    view = directory.parent / f"fid-n{count}-seeds-0-{count - 1}"
                    for new_seeds in seed_ranges(missing):
                        print("PLAN " + shlex.join(generate_command(row, directory, new_seeds, args)))
                    available.update(range(count))
                    print(f"PLAN FID view={view}; ref={args.fid_ref}; num={count}")
                    if view.exists():
                        ensure_protocol(view, {"pool_protocol_id": digest(value), "num_images": count}, create=False)
                        inventory(view, range(count), complete=True)
                    if args.verify_only and any(item["condition"] == row["condition"] and int(item["num_images"]) == count for item in results):
                        inventory(view, range(count), complete=True)
                        if not set(range(count)).issubset(images):
                            raise ValueError("Completed evaluation has missing pool images")
                        if any(item["protocol_id"] != digest(value) for item in results
                               if item["condition"] == row["condition"] and int(item["num_images"]) == count):
                            raise ValueError("Completed evaluation protocol changed")
        if not args.samples_only:
            print(f"PLAN total new_images={total_new}; pending FID evaluations={pending_evaluations}; "
                  f"conditions={len(rows)}; stages={stages}")
        return
    if os.environ.get("EDM_GPU_WAIT_DONE") != "1":
        raise ValueError("Use scripts/run.sh for GPU work; it calls the shared GPU waiter")
    # Revalidate local policy even when this module is invoked directly.
    configured = subprocess.check_output(
        ["bash", "-c", 'source scripts/load_gpu_config.sh || exit 1; printf "%s" "$GPU_INDEX"'],
        cwd=REPO_ROOT, text=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != configured or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("CUDA settings must match the local GPU configuration")
    if shutil.which(args.torchrun) is None:
        raise ValueError(f"Missing torchrun: {args.torchrun}")
    stop = args.exp_root / "STOP_AFTER_CURRENT_CONDITION"
    if stop.exists():
        print(f"Stop file exists: {stop}; no GPU command started")
        return
    if not args.resume and ((args.exp_root / "images").exists() or results):
        raise ValueError("Run data exists; use --resume or a new EXP_ROOT")
    (args.exp_root / "logs").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DNNLIB_CACHE_DIR", str(REPO_ROOT / ".cache/dnnlib"))
    with (args.exp_root / "logs/.run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another experiment 8-10 process is using this EXP_ROOT") from error
        results = load_results(args.exp_root)
        for count in ([0] if args.samples_only else stages):
            for row in rows:
                if stop.exists():
                    print("Stop file found; exiting before the next condition")
                    return
                if args.samples_only:
                    directory = args.exp_root / "images" / row["condition"] / f"samples-seeds-{args.sample_seeds}"
                    ensure_protocol(directory, protocol(row, args, args.sample_seeds))
                    existing = inventory(directory, sample_seeds)
                    for seeds in seed_ranges(sorted(set(sample_seeds) - set(existing))):
                        run_logged(generate_command(row, directory, seeds, args), args.exp_root / "logs" / f"{row['condition']}-samples.log")
                    inventory(directory, sample_seeds, complete=True)
                    print(f"SAMPLES {row['condition']}: {len(sample_seeds)} verified", flush=True)
                else:
                    evaluate(row, count, args, results)
        if not args.samples_only:
            aggregate(args.exp_root, results)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, SyntaxError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
