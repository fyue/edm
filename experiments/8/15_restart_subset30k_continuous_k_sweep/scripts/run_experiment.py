#!/usr/bin/env python3
"""Experiment 8-15 evaluation engine: continuous subset30k, direct FID-50k.

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
import queue
import signal
import threading
import shlex
import shutil
import statistics
import subprocess
import sys
import time



SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from condition_pause import ConditionPause, StopRequested

DEFAULT_ROOT = SCRIPT_ROOT.parent
MAX_IMAGES = 50000
STAGES = (50000,)
MODE = os.environ.get("FIXED_MODE", "main18")
if MODE not in ("main18", "nfe55"):
    raise ValueError("FIXED_MODE must be main18 or nfe55")
CONFIG_NAME = f"restart_{MODE}.tsv"
EXPERIMENT_ID = f"8-15-{MODE}"
CHECKPOINTS = ("20k", "40k", "60k", "80k", "100k")
KS = tuple(range(9))

FIELDS = [
    "condition", "checkpoint", "main_steps", "restart_steps", "restart_k",
    "t_min", "t_max", "nfe", "actual_t_min", "restart_index", "num_images", "seeds", "fid", "generate_sec",
    "fid_sec", "reused_images", "reuse_source", "network", "images", "log",
    "fid_ref", "gen_batch", "fid_batch", "rho", "restart_noise", "protocol_id",
    "protocol", "gpu_name", "generated_images", "elapsed_sec",
]

IDENTITIES = {}
TERMINATE_GRACE_SEC = 10


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
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
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
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in IDENTITIES:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        IDENTITIES[key] = {"size": stat.st_size, "sha256": hasher.hexdigest()}
    return dict(IDENTITIES[key])


def schedule_point(main_steps):
    # EDMPrecond.round_sigma is identity for these checkpoints.
    times = [(80 ** (1 / 7) + i / (main_steps - 1)
              * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
             for i in range(main_steps)] + [0.0]
    index = min(range(len(times)), key=lambda i: abs(times[i] - 0.14))
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
            raise ValueError("Require 20k,40k,60k,80k,100k x K=0..8")
        if row["condition"] != f"{row['checkpoint']}-k{k}":
            raise ValueError("Condition name must be <checkpoint>-k<K>")
        seen.add(pair)
        names.add(row["condition"])
        target = (18 if MODE == "main18" else 28 - 2 * k, 3 if k else 0, 0.14 if k else 0, 0.30 if k else 0, 35 + 4 * k if MODE == "main18" else 55)
        if tuple(row[field] for field in ("main_steps", "restart_steps", "t_min", "t_max", "nfe")) != target:
            raise ValueError("Invalid fixed sampler configuration")
        if 2 * row["main_steps"] - 1 + 4 * k != row["nfe"]:
            raise ValueError("NFE mismatch")
        row["actual_t_min"], row["restart_index"] = schedule_point(row["main_steps"]) if k else (None, None)
        row["network"] = str((REPO_ROOT / row["network"]).resolve())
        if Path(row["network"]).name != f"network-snapshot-{int(row['checkpoint'][:-1]) * 1000:06d}.pkl":
            raise ValueError("Checkpoint label and snapshot disagree")
    if seen != expected:
        raise ValueError("Incomplete 45-condition configuration")
    for checkpoint in CHECKPOINTS:
        if len({row["network"] for row in rows if row["checkpoint"] == checkpoint}) != 1:
            raise ValueError("All K values must use the same checkpoint")
    return rows


def parse_checkpoints(value):
    tokens = value.replace(",", " ").split()
    if not tokens or tokens == ["all"]:
        return set(CHECKPOINTS)
    wanted = set()
    for token in tokens:
        match = re.fullmatch(r"(\d+)k(?:-(\d+)k(?::(\d+)k)?)?", token)
        if match is None:
            raise ValueError(f"Invalid checkpoint selector: {token}")
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        interval = int(match[3]) if match[3] else 20
        if f"{start}k" not in CHECKPOINTS or f"{end}k" not in CHECKPOINTS:
            raise ValueError(f"Checkpoint outside 20k,40k,60k,80k,100k: {token}")
        if start > end or interval < 20 or interval % 20 or (end - start) % interval:
            raise ValueError(f"Invalid checkpoint range/interval: {token}")
        selected = {f"{step}k" for step in range(start, end + 1, interval)}
        if not selected.issubset(CHECKPOINTS):
            raise ValueError(f"Checkpoint outside 20k,40k,60k,80k,100k: {token}")
        wanted.update(selected)
    return wanted


def select_conditions(rows, checkpoints, conditions):
    # Validate selection before reading any model: another PC's files may be absent.
    selectors = (("checkpoint", parse_checkpoints(checkpoints)),
                 ("condition", set(conditions.replace(",", " ").split())))
    for field, wanted in selectors:
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
    # Content identities keep protocols portable across PCs and file mtimes.
    value = {"experiment": EXPERIMENT_ID, "condition": {key: item for key, item in row.items() if key != "network"},
            "network": identity(row["network"]),
            "fid_ref": identity(args.fid_ref), "rho": 7, "restart_noise": 1,
            "sigma_min": 0.002, "sigma_max": 80, "S_churn": 0,
            "gen_batch": args.gen_batch, "fid_batch": args.fid_batch,
            "seeds": seeds or "0-49999", "generate": identity(REPO_ROOT / "generate.py"),
            "fid": identity(REPO_ROOT / "fid.py"), "runner": identity(Path(__file__))}
    value["watcher"] = identity(SCRIPT_ROOT / "watch_checkpoints.py")
    value["condition_pause"] = identity(REPO_ROOT / "scripts/condition_pause.py")
    value["gpu_wait"] = identity(REPO_ROOT / "scripts/wait_for_gpu_and_run.sh")
    value["gpu_config"] = identity(REPO_ROOT / "scripts/load_gpu_config.sh")
    value["pause_eta"] = identity(REPO_ROOT / "scripts/pause_eta.py")
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
               "--rdzv_id=experiment8-15", "--nproc_per_node=1", "generate.py",
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


def phase(args, name, log=None):
    controller = getattr(args, "pause", None)
    if controller is not None:
        controller.phase(name, log=log)


def terminate_group(process):
    # torchrun owns workers too; do not leave a GPU worker running after its leader exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + TERMINATE_GRACE_SEC
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_logged(command, log):
    print(shlex.join(command), flush=True)
    started, scalar = time.monotonic(), None
    lines = queue.Queue()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)

        def read_output():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        ended = False
        try:
            while not ended or process.poll() is None:
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    ended = True
                else:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line.strip()):
                        scalar = float(line.strip())
            code = process.wait()
        finally:
            if process.poll() is None:
                terminate_group(process)
            reader.join(timeout=1)
            process.stdout.close()
        elapsed = time.monotonic() - started
        if code:
            raise RuntimeError(f"Command failed; see {log}")
    return elapsed, scalar


def recover_staging(staging, pool, allowed):
    allowed = set(allowed)
    saved = 0
    for path in staging.rglob("*.png"):
        if not path.stem.isdigit():
            raise ValueError(f"Unexpected staging image: {path}")
        seed = int(path.stem)
        if seed not in allowed or path.relative_to(staging) != image_relative(seed):
            raise ValueError(f"Unexpected staging seed/path: {path}")
        try:
            validate_png(path)
        except (ValueError, OSError, SyntaxError):
            path.unlink()
            continue
        target = pool / image_relative(seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publication without overwriting an existing image, on the same filesystem.
        try:
            os.link(path, target)
            saved += 1
        except FileExistsError:
            validate_png(target)
            if path.read_bytes() != target.read_bytes():
                raise ValueError(f"Staging image differs from saved image: {target}")
        path.unlink()
    return saved


def progress_record(path, value):
    if path.exists():
        progress = json.loads(path.read_text())
        if progress["protocol_id"] != digest(value):
            raise ValueError("Progress protocol mismatch")
        return progress
    return {"protocol_id": digest(value), "generate_sec": 0.0, "fid_sec": 0.0,
            "generated_images": 0, "elapsed_sec": 0.0}


def generate_missing(row, pool, wanted, args, log, progress, progress_path, value):
    staging = pool.parent / (pool.name + "-staging")
    ensure_protocol(staging, value)
    saved = recover_staging(staging, pool, wanted)
    progress["generated_images"] += saved
    write_json(progress_path, progress)
    existing = inventory(pool, range(MAX_IMAGES))
    missing = sorted(set(wanted) - set(existing))
    for seeds in seed_ranges(missing):
        started = time.monotonic()
        try:
            phase(args, "generate", log)
            elapsed, _ = run_logged(generate_command(row, staging, seeds, args), log)
        except BaseException:
            elapsed = time.monotonic() - started
            progress["generate_sec"] += elapsed
            progress["elapsed_sec"] += elapsed
            progress["generated_images"] += recover_staging(staging, pool, wanted)
            write_json(progress_path, progress)
            raise
        progress["elapsed_sec"] += elapsed
        progress["generate_sec"] += elapsed
        progress["generated_images"] += recover_staging(staging, pool, wanted)
        write_json(progress_path, progress)
    phase(args, "validate", log)
    inventory(pool, wanted, complete=True)


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
    validate_results(results)
    groups = comparison_rows(results)
    destination = root / "results"
    results = sorted(results, key=lambda row: (int(row["num_images"]), int(row["checkpoint"][:-1]), int(row["restart_k"])))
    write_tsv(destination / "restart_fid_summary.tsv",
              [{key: value for key, value in row.items() if key != "protocol"} for row in results],
              [field for field in FIELDS if field != "protocol"])
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
        axes[0].set(xlabel="K", ylabel="FID", title=f"FID-{count}, {MODE}, subset30k, t_min=0.14")
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
        axes[2].set(xlabel="Training (kimg)", ylabel="Optimal K", title="Best among K=0..8")
        axes[2].set_yticks(KS)
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(destination / f"fixed_{MODE}_tmin014_fid{count}.{extension}", dpi=180)
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
    previous = 0  # Direct FID-50k; no smaller evaluation is required.
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
    progress_path = args.exp_root / "progress" / f"{row['condition']}-fid{count}.json"
    progress = progress_record(progress_path, value)
    staging = pool.parent / (pool.name + "-staging")
    if staging.exists():
        ensure_protocol(staging, value, create=False)
        progress["generated_images"] += recover_staging(staging, pool, range(MAX_IMAGES))
        write_json(progress_path, progress)
    log = args.exp_root / "logs" / f"{row['condition']}-fid{count}.log"
    generate_missing(row, pool, range(count), args, log, progress, progress_path, value)
    generation_sec = progress["generate_sec"]
    existing = inventory(pool, range(MAX_IMAGES))
    if not set(range(count)).issubset(existing):
        raise ValueError("Generation left missing seeds")
    view_started = time.monotonic()
    ensure_protocol(view, {"pool_protocol_id": run_id, "num_images": count})
    for seed in range(count):
        link_image(existing[seed], view / image_relative(seed))
    inventory(view, range(count), complete=True)
    progress["elapsed_sec"] += time.monotonic() - view_started
    write_json(progress_path, progress)
    command = [args.torchrun, "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
               "--rdzv_id=experiment8-15", "--nproc_per_node=1", "fid.py", "calc",
               f"--images={view}", f"--ref={args.fid_ref}", f"--num={count}", f"--batch={args.fid_batch}"]
    # A pause request must allow generation AND this FID to finish together.
    phase(args, "fid", log)
    fid_sec, fid = run_logged(command, log)
    progress["fid_sec"] += fid_sec
    progress["elapsed_sec"] += fid_sec
    write_json(progress_path, progress)
    fid_sec = progress["fid_sec"]
    if fid is None or not math.isfinite(fid) or fid < 0:
        raise ValueError(f"Could not parse finite FID from {log}")
    phase(args, "save", log)
    result = dict(row, num_images=count, seeds=f"0-{count - 1}", fid=fid,
                  generate_sec=generation_sec, fid_sec=fid_sec,
                  reused_images=0, reuse_source="",
                  images=str(view), log=str(log), fid_ref=str(args.fid_ref),
                  gen_batch=args.gen_batch, fid_batch=args.fid_batch, rho=7,
                  restart_noise=1, protocol_id=run_id, protocol=value,
                  gpu_name=args.gpu_name, generated_images=min(count, progress["generated_images"]),
                  elapsed_sec=progress["elapsed_sec"])
    results.append(result)
    # Per-result JSON is the completion marker; interrupted TSV/plot updates recover from it.
    write_json(args.exp_root / "results" / f"{row['condition']}-fid{count}.json", result)
    aggregate(args.exp_root, results)


def run_condition(row, count, args, results, pause):
    """Use the same condition-boundary controller as experiment 8-13."""
    seeds = args.sample_seeds if args.samples_only else f"0-{count - 1}"
    pause.info["num_images"] = len(parse_seeds(seeds)) if args.samples_only else count
    pause.boundary(row, seeds)
    stat = Path(row['network']).stat()
    current = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    if current != args.snapshot_signature:
        raise ValueError('Snapshot changed after CPU validation; restart after writing finishes')
    args.gpu_name = os.environ.get("EXPECTED_GPU_NAME", getattr(args, "gpu_name", "unknown"))
    args.pause = pause
    log = args.exp_root / "logs" / (f"{row['condition']}-samples.log" if args.samples_only
                                   else f"{row['condition']}-fid{count}.log")
    pause.phase("validate", row, seeds, log)
    if args.samples_only:
        sample_seeds = parse_seeds(args.sample_seeds)
        directory = args.exp_root / "images" / row["condition"] / f"samples-seeds-{args.sample_seeds}"
        ensure_protocol(directory, protocol(row, args, args.sample_seeds))
        sample_value = protocol(row, args, args.sample_seeds)
        progress_path = args.exp_root / "progress" / f"{row['condition']}-samples.json"
        progress = progress_record(progress_path, sample_value)
        generate_missing(row, directory, sample_seeds, args, log, progress, progress_path, sample_value)
        pause.phase("validate")
        inventory(directory, sample_seeds, complete=True)
        print(f"SAMPLES {row['condition']}: {len(sample_seeds)} verified", flush=True)
    else:
        evaluate(row, count, args, results)
    pause.phase("saved")


def validate_results(results):
    definitions = {row["condition"]: {key: value for key, value in row.items() if key != "network"}
                   for row in load_conditions(DEFAULT_ROOT / "config" / CONFIG_NAME)}
    shared, networks = None, {}
    for row in results:
        if set(row) != set(FIELDS):
            raise ValueError("Invalid completed result fields")
        count = row["num_images"]
        value = row["protocol"]
        definition = definitions.get(row["condition"])
        if (count not in STAGES or definition is None or row["seeds"] != f"0-{count - 1}"
                or not math.isfinite(float(row["fid"])) or float(row["fid"]) < 0):
            raise ValueError("Invalid completed result condition/stage/FID/seeds")
        if (not isinstance(value, dict) or value.get("experiment") != EXPERIMENT_ID
                or value.get("condition") != definition or value.get("seeds") != "0-49999"
                or row["protocol_id"] != digest(value)
                or any(row[key] != item for key, item in definition.items())):
            raise ValueError("Completed result protocol or sampling conditions mismatch")
        expected_keys = {"experiment", "condition", "network", "fid_ref", "rho", "restart_noise",
                         "sigma_min", "sigma_max", "S_churn", "gen_batch", "fid_batch", "seeds",
                         "generate", "fid", "runner", "watcher", "condition_pause", "gpu_wait", "gpu_config", "pause_eta"}
        if set(value) != expected_keys:
            raise ValueError("Incomplete reproducibility protocol")
        for key in ("network", "fid_ref", "generate", "fid", "runner", "watcher", "condition_pause", "gpu_wait", "gpu_config", "pause_eta"):
            item = value[key]
            if (not isinstance(item, dict) or set(item) != {"size", "sha256"}
                    or not isinstance(item["size"], int) or item["size"] <= 0
                    or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
                raise ValueError(f"Invalid SHA256 identity: {key}")
        for key, expected in (("rho", 7), ("restart_noise", 1), ("sigma_min", 0.002),
                              ("sigma_max", 80), ("S_churn", 0)):
            if value[key] != expected:
                raise ValueError(f"Invalid protocol parameter: {key}")
        for key in ("gen_batch", "fid_batch", "rho", "restart_noise"):
            if row[key] != value[key] or not isinstance(value[key], int) or value[key] <= 0:
                raise ValueError(f"Result/protocol mismatch: {key}")
        for key in ("generate_sec", "fid_sec", "elapsed_sec"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError(f"Invalid measured time: {key}")
        if (not isinstance(row["generated_images"], int) or not 0 <= row["generated_images"] <= count
                or not isinstance(row["gpu_name"], str) or not row["gpu_name"]
                or row["reused_images"] != 0 or row["reuse_source"] != ""):
            raise ValueError("Invalid generation/GPU metadata")
        common = {key: item for key, item in value.items() if key not in ("condition", "network")}
        if shared is not None and shared != common:
            raise ValueError("Result roots disagree on batch, code or FID reference SHA256")
        shared = common
        checkpoint = row["checkpoint"]
        if checkpoint in networks and networks[checkpoint] != value["network"]:
            raise ValueError(f"Model SHA256 mismatch for {checkpoint}")
        networks[checkpoint] = value["network"]
    comparison_rows(results)


def load_results(root):
    results = []
    for path in sorted((root / "results").glob("*-fid[0-9]*.json")):
        row = json.loads(path.read_text())
        if path.name != f"{row.get('condition')}-fid{row.get('num_images')}.json":
            raise ValueError(f"Invalid completion marker filename: {path}")
        results.append(row)
    validate_results(results)
    return results


def merged_results(roots):
    for root in roots:
        if not (root / "results").is_dir():
            raise ValueError(f"Missing result directory: {root / 'results'}")
    results = [row for root in roots for row in load_results(root)]
    validate_results(results)
    return results


