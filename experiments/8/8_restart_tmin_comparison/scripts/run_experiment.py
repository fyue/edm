#!/usr/bin/env python3
"""Experiment 8-8: K=5 Restart t_min comparison, nested FID-5k/25k/50k.

GPU runs must enter through run.sh (which uses the shared GPU waiter).
CLI options override their equivalent environment variables.
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
SOURCE_ROOT = REPO_ROOT / "experiments/8/4_restart_k_sweep_10k_interval"
PROGRESSIVE_SOURCE_ROOT = REPO_ROOT / "experiments/8/6_exact_10k_restart_k5_fid50k_1seed"
MAX_IMAGES = 50000
STAGES = (5000, 25000, 50000)
CHECKPOINTS = ("20k", "40k", "60k", "170k")


def checkpoint_label(checkpoint):
    return "170k best (Exp.3-10)" if checkpoint == "170k" else checkpoint
FIELDS = [
    "condition", "checkpoint", "main_steps", "restart_steps", "restart_k",
    "t_min", "t_max", "nfe", "num_images", "seeds", "fid", "generate_sec",
    "fid_sec", "reused_images", "reuse_source", "network", "images", "log",
    "fid_ref", "gen_batch", "fid_batch", "rho", "restart_noise", "protocol_id",
]
COMPARISON_FIELDS = ["checkpoint", "num_images", "seeds", "fid_006", "fid_014", "delta"]


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


def load_conditions(path):
    rows = read_tsv(path)
    expected = {(checkpoint, t_min) for checkpoint in CHECKPOINTS for t_min in (0.06, 0.14)}
    seen, names = set(), set()
    for row in rows:
        for field in ("main_steps", "restart_steps", "restart_k", "nfe"):
            row[field] = int(row[field])
        row["t_min"], row["t_max"] = float(row["t_min"]), float(row["t_max"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", row["condition"]):
            raise ValueError("Unsafe condition name")
        pair = row["checkpoint"], row["t_min"]
        if pair not in expected or pair in seen or row["condition"] in names:
            raise ValueError("Require exactly the eight 20k/40k/60k/170k x 0.06/0.14 conditions")
        seen.add(pair)
        names.add(row["condition"])
        if (row["main_steps"], row["restart_steps"], row["restart_k"], row["t_max"], row["nfe"]) != (18, 3, 5, 0.30, 55):
            raise ValueError("Require main=18, Restart steps=3, K=5, t_max=0.30, NFE=55")
        row["network"] = str((REPO_ROOT / row["network"]).resolve())
        identity(row["network"])
    if seen != expected:
        raise ValueError("Incomplete eight-condition config")
    for checkpoint in CHECKPOINTS:
        if len({row["network"] for row in rows if row["checkpoint"] == checkpoint}) != 1:
            raise ValueError(f"{checkpoint}: both t_min conditions must use the same checkpoint")
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


def source_candidate(row, source_root, gen_batch):
    """Legacy 8-4/8-6: validate recorded settings, then salvage valid seed PNGs."""
    if row["t_min"] != 0.06 or gen_batch != 64:
        return None, "not a legacy 0.06/batch64 condition"
    name = f"{row['checkpoint']}-restart-k5"
    matches = [item for item in read_tsv(source_root / "results/restart_fid_summary.tsv")
               if item.get("condition") == name and item.get("num_images") in {str(stage) for stage in STAGES}
               and item.get("seeds") == f"0-{int(item['num_images']) - 1}"]
    if not matches:
        return None, "missing legacy summary row"
    matches.sort(key=lambda item: int(item["num_images"]), reverse=True)
    if len({item["num_images"] for item in matches}) != len(matches):
        return None, "ambiguous legacy summary rows"
    source = dict(matches[0])
    try:
        for field in ("main_steps", "restart_steps", "restart_k", "nfe", "t_min", "t_max"):
            if float(source[field]) != row[field]:
                return None, f"legacy {field} mismatch"
        if source.get("method") != "restart" or source.get("checkpoint") != row["checkpoint"]:
            return None, "legacy checkpoint/method mismatch"
        if identity(REPO_ROOT / source["network"]) != identity(row["network"]):
            return None, "legacy checkpoint file mismatch"
        config = next((source_root / "config" / filename for filename in
                       ("restart_k5_fid50k.tsv", "restart_k0_8_fid5k.tsv")
                       if (source_root / "config" / filename).exists()), None)
        if config is None:
            return None, "missing legacy config"
        with config.open(encoding="utf-8") as handle:
            configs = [line.rstrip("\n").split("\t") for line in handle if line.startswith(name + "\t")]
        if len(configs) != 1 or len(configs[0]) != 10:
            return None, "missing legacy config row"
        values = configs[0]
        if values[1] != row["checkpoint"] or any(float(value) != row[field] for value, field in
                zip(values[2:8], ("main_steps", "restart_steps", "restart_k", "t_min", "t_max", "nfe"))):
            return None, "legacy config settings mismatch"
        if identity(REPO_ROOT / values[9]) != identity(row["network"]):
            return None, "legacy config network mismatch"
        directory = (REPO_ROOT / source["images"]).resolve()
        directory.relative_to((source_root / "images").resolve())
        valid = []
        for seed in range(int(source["num_images"])):
            path = directory / image_relative(seed)
            try:
                validate_png(path)
            except (OSError, ValueError, SyntaxError):
                continue
            valid.append(seed)
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        return None, f"legacy validation failed: {error}"
    if not valid:
        return None, "no valid legacy PNGs"
    source["_valid_seeds"] = valid
    return source, f"validated {len(valid)}/{source['num_images']} RGB seed PNGs; legacy rho=7/noise=1/batch64 defaults"


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
            "fid": identity(REPO_ROOT / "fid.py")}
    directory = "pool" if seeds is None else f"samples-seeds-{seeds}"
    path = args.exp_root / "images" / row["condition"] / directory / ".protocol.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        relocated = json.loads(json.dumps(recorded))
        # Preserve legacy IDs while comparing repo-relative paths across synced PCs.
        old_root = Path(recorded["generate"]["path"]).parent
        try:
            for key in ("network", "fid_ref", "generate", "fid"):
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


def ensure_extension(pool, value, create=True):
    """Keep the legacy base ID; only the additional seed domain gets new metadata."""
    expected = {"base_protocol_id": digest(value), "base_seeds": "0-24999",
                "additional_seeds": "25000-49999", "max_images": MAX_IMAGES}
    path = pool / ".extension.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Extension metadata mismatch: {pool}")
    elif create:
        write_json(path, expected)
    else:
        raise ValueError(f"Missing 50k extension metadata: {pool}")


def reused_provenance(pool, count):
    seeds, paths = set(), []
    legacy = pool / ".reuse.json"
    if legacy.exists():
        paths.append(json.loads(legacy.read_text())["source"]["images"])
        seeds.update(range(min(5000, count)))
    metadata = pool / ".imports.json"
    if metadata.exists():
        for entry in json.loads(metadata.read_text()):
            paths.append(entry["images"])
            for value in entry["seed_ranges"]:
                seeds.update(seed for seed in parse_seeds(value) if seed < count)
    return len(seeds), ";".join(dict.fromkeys(paths))


def generate_command(row, directory, seeds, args):
    info = json.dumps({"0": [3, 5, row["t_min"], 0.30]}, separators=(",", ":"))
    return [args.torchrun, "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
            "--rdzv_id=experiment8-8", "--nproc_per_node=1", "generate.py",
            f"--network={row['network']}", f"--outdir={directory}", f"--seeds={seeds}",
            f"--batch={args.gen_batch}", "--sampler=restart", "--steps=18", "--rho=7",
            "--sigma_min=0.002", "--sigma_max=80", "--S_churn=0",
            f"--restart_info={info}", "--restart_noise=1", "--subdirs"]


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
    for item in results:
        key = item["checkpoint"], int(item["num_images"]), item["seeds"]
        pair = groups.setdefault(key, {})
        t_min = float(item["t_min"])
        if t_min in pair:
            raise ValueError(f"Duplicate result: {key}, t_min={t_min}")
        pair[t_min] = float(item["fid"])
    output = []
    for (checkpoint, count, seeds), pair in sorted(groups.items(), key=lambda item: (int(item[0][0].rstrip("k")), item[0][1], item[0][2])):
        low, high = pair.get(0.06, "NA"), pair.get(0.14, "NA")
        output.append(dict(zip(COMPARISON_FIELDS, [checkpoint, count, seeds, low, high,
                                                  low - high if low != "NA" and high != "NA" else "NA"])))
    return output


def aggregate(root, results):
    destination = root / "results"
    comparisons = comparison_rows(results)
    write_tsv(destination / "restart_fid_summary.tsv", results, FIELDS)
    write_json(destination / "restart_fid_summary.json", results)
    write_tsv(destination / "tmin_comparison.tsv", comparisons, COMPARISON_FIELDS)
    write_json(destination / "tmin_comparison.json", comparisons)
    complete = [row for row in comparisons if row["delta"] != "NA"]
    if not complete:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for count in sorted({row["num_images"] for row in complete}):
        rows = [row for row in complete if row["num_images"] == count]
        figure, (bars, difference) = plt.subplots(1, 2, figsize=(9, 3.8))
        x = list(range(len(rows)))
        bars.bar([value - 0.18 for value in x], [row["fid_006"] for row in rows], width=0.36, label="t_min=0.06")
        bars.bar([value + 0.18 for value in x], [row["fid_014"] for row in rows], width=0.36, label="t_min=0.14")
        bars.set_ylabel("FID (lower is better)")
        bars.legend()
        difference.bar(x, [row["delta"] for row in rows], color=["tab:green" if row["delta"] >= 0 else "tab:red" for row in rows])
        difference.axhline(0, color="black", linewidth=0.8)
        difference.set_ylabel("FID(0.06) - FID(0.14)")
        difference.set_title("Positive favors t_min=0.14")
        for axis in (bars, difference):
            axis.set_xticks(x)
            axis.set_xticklabels([checkpoint_label(row["checkpoint"]) for row in rows])
            axis.set_xlabel("Training checkpoint")
        figure.suptitle(f"Experiment 8-8: Restart K=5, NFE=55, FID-{count:,}, seeds 0-{count - 1}")
        if any(row["checkpoint"] == "170k" for row in rows):
            figure.text(0.5, 0.01, "170k: different training branch (Exp.3-10)", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.05, 1, 1))
        for extension in ("png", "svg"):
            figure.savefig(destination / f"restart_tmin_fid{count}.{extension}", dpi=180)
        plt.close(figure)

        ordered = sorted(rows, key=lambda row: int(row["checkpoint"].rstrip("k")))
        checkpoints = [int(row["checkpoint"].rstrip("k")) for row in ordered]
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for field, other, label, color, marker in (
            ("fid_006", "fid_014", "t_min=0.06", "tab:blue", "o"),
            ("fid_014", "fid_006", "t_min=0.14", "tab:orange", "s"),
        ):
            values = [row[field] for row in ordered]
            axis.plot(checkpoints, values, marker=marker, color=color,
                      linewidth=2, markersize=7, label=label)
            for checkpoint, fid, row in zip(checkpoints, values, ordered):
                offset = 10 if fid > row[other] or (fid == row[other] and field == "fid_006") else -15
                axis.annotate(f"{fid:.5f}", (checkpoint, fid), xytext=(0, offset),
                              textcoords="offset points", ha="center", color=color, fontsize=9)
        axis.set_xticks(checkpoints)
        axis.set_xticklabels([checkpoint_label(row["checkpoint"]) for row in ordered])
        axis.set_xlabel("Training checkpoint")
        axis.set_ylabel("FID (lower is better)")
        axis.set_title(f"Restart K=5, NFE=55, FID-{count:,}\nSeeds 0-{count - 1}")
        axis.margins(x=0.15, y=0.2)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        if any(row["checkpoint"] == "170k" for row in ordered):
            figure.text(0.5, 0.01, "170k: different training branch; connecting lines are comparison guides, not a training trajectory.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.05, 1, 1))
        for extension in ("png", "svg"):
            figure.savefig(destination / f"restart_tmin_fid{count}_lines.{extension}", dpi=180)
        plt.close(figure)


def prepare_pool(row, args, value, source):
    pool = args.exp_root / "images" / row["condition"] / "pool"
    ensure_protocol(pool, value)
    existing = inventory(pool, range(MAX_IMAGES))
    if any(seed >= 25000 for seed in existing):
        ensure_extension(pool, value, create=False)
    metadata = pool / ".imports.json"
    imports = json.loads(metadata.read_text()) if metadata.exists() else []
    for candidate in source or []:
        missing = sorted(set(candidate["_valid_seeds"]) - set(existing))
        if not missing:
            continue
        if any(seed >= 25000 for seed in missing):
            ensure_extension(pool, value)
        # Persist intent first; after an interruption verify the intended source again.
        entry = {"images": candidate["images"], "network": candidate["network"],
                 "seed_ranges": seed_ranges(missing),
                 "validation": "legacy summary/config, RGB PNG; rho=7/noise=1/batch64 defaults"}
        if entry not in imports:
            imports.append(entry)
            write_json(metadata, imports)
        for seed in missing:
            target = pool / image_relative(seed)
            link_image(REPO_ROOT / candidate["images"] / image_relative(seed), target)
            existing[seed] = target
        print(f"REUSED {row['condition']}: {len(missing)} images from {candidate['images']}", flush=True)
    return pool


def evaluate(row, count, args, results, source):
    value = protocol(row, args)
    pool = prepare_pool(row, args, value, source)
    run_id = digest(value)
    recorded = [item for item in results if item["condition"] == row["condition"] and int(item["num_images"]) == count]
    if len(recorded) > 1:
        raise ValueError("Duplicate summary result")
    view = pool.parent / f"fid-n{count}-seeds-0-{count - 1}"
    existing = inventory(pool, range(MAX_IMAGES))
    previous = {5000: 0, 25000: 5000, 50000: 25000}[count]
    if not set(range(previous)).issubset(existing):
        raise ValueError(f"{count} extension requires all first {previous} images")
    if count == 50000:
        ensure_extension(pool, value)
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
               "--rdzv_id=experiment8-8", "--nproc_per_node=1", "fid.py", "calc",
               f"--images={view}", f"--ref={args.fid_ref}", f"--num={count}", f"--batch={args.fid_batch}"]
    fid_sec, fid = run_logged(command, log)
    if fid is None or not math.isfinite(fid):
        raise ValueError(f"Could not parse finite FID from {log}")
    reused_images, reuse_source = reused_provenance(pool, count)
    result = dict(row, num_images=count, seeds=f"0-{count - 1}", fid=fid,
                  generate_sec=generation_sec, fid_sec=fid_sec,
                  reused_images=reused_images, reuse_source=reuse_source,
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
    parser.add_argument("--fid-stage", choices=("5000", "25000", "50000", "all"), default=os.environ.get("FID_STAGE", "5000"),
                        help="run prerequisites in order; 50000/all extends pools to 50k")
    parser.add_argument("--checkpoint-filter", default=os.environ.get("CHECKPOINT_FILTER", ""))
    parser.add_argument("--condition-filter", default=os.environ.get("CONDITION_FILTER", ""))
    parser.add_argument("--exp-root", type=Path, default=Path(os.environ.get("EXP_ROOT", str(DEFAULT_ROOT))))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("CONFIG", str(DEFAULT_ROOT / "config/restart_tmin.tsv"))))
    parser.add_argument("--fid-ref", type=Path, default=Path(os.environ.get("FID_REF", "fid-refs/cifar10-32x32.npz")))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--progressive-source-root", type=Path, default=PROGRESSIVE_SOURCE_ROOT)
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
    if args.fid_stage not in ("5000", "25000", "50000", "all"):
        parser.error("invalid FID_STAGE")
    return args


def main():
    os.chdir(REPO_ROOT)
    args = arguments()
    args.exp_root = args.exp_root.resolve()
    args.fid_ref = args.fid_ref.resolve()
    args.source_root = args.source_root.resolve()
    args.progressive_source_root = args.progressive_source_root.resolve()
    rows = select_conditions(load_conditions(args.config), args.checkpoint_filter, args.condition_filter)
    identity(args.fid_ref)
    results = load_results(args.exp_root)
    if args.aggregate_only:
        aggregate(args.exp_root, results)
        print(f"Aggregated {len(results)} completed evaluations")
        return
    sources = {}
    for row in rows:
        candidates = []
        if not args.samples_only:
            for source_root in dict.fromkeys((args.progressive_source_root, args.source_root)):
                source, reason = source_candidate(row, source_root, args.gen_batch)
                if source is not None:
                    candidates.append(source)
                print(f"REUSE {row['condition']} [{source_root.name}]: {reason}", flush=True)
        sources[row["condition"]] = candidates
    maximum = 50000 if args.fid_stage == "all" else int(args.fid_stage)
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
            if (directory / ".extension.json").exists() or any(seed >= 25000 for seed in images):
                ensure_extension(directory, value, create=False)
            print(f"PLAN {row['condition']}: checkpoint={row['checkpoint']} t_min={row['t_min']} t_max=0.30 "
                  f"main=18 restart_steps=3 K=5 NFE=55 pool={directory} existing={len(images)}")
            if args.samples_only:
                sample_dir = args.exp_root / "images" / row["condition"] / f"samples-seeds-{args.sample_seeds}"
                ensure_protocol(sample_dir, protocol(row, args, args.sample_seeds), create=False)
                sample_images = inventory(sample_dir, sample_seeds, complete=args.verify_only)
                print(f"SAMPLES existing={len(sample_images)}/{len(sample_seeds)}; directory={sample_dir}")
                print("PLAN " + shlex.join(generate_command(row, sample_dir, args.sample_seeds, args)))
            else:
                available = set(images)
                for source in sources[row["condition"]]:
                    available.update(source["_valid_seeds"])
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
            raise ValueError("Another experiment 8-8 process is using this EXP_ROOT") from error
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
                    evaluate(row, count, args, results, sources[row["condition"]])
        if not args.samples_only:
            aggregate(args.exp_root, results)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, SyntaxError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
