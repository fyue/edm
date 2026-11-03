"""Explicit, output-local approval of the reviewed pause-only runner change."""
import ast
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


LEGACY = {"sha256": "a5272ce13e72593fe6f3fda2485e1ab043e228606a6b91d59043b201be5bf443", "size": 45568}
DIRECTORY = "runner_compatibility"
STABLE_FUNCTIONS = (
    "schedule_point", "load_conditions", "parse_seeds", "image_relative",
    "validate_png", "inventory", "link_image", "generate_command", "seed_ranges",
    "recover_staging", "validate_results",
)


def content_identity(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def files(module):
    return {
        "runner": Path(module.__file__).resolve(),
        "compatibility": Path(__file__).resolve(),
        "condition_pause": module.REPO_ROOT / "scripts/condition_pause.py",
        "pause_eta": module.REPO_ROOT / "scripts/pause_eta.py",
        "gpu_waiter": module.REPO_ROOT / "scripts/wait_for_gpu_and_run.sh",
        "gpu_config_loader": module.REPO_ROOT / "scripts/load_gpu_config.sh",
    }


def read_registration(root, module):
    directory = root / DIRECTORY
    path = directory / "registration.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record.get("schema") != 1 or record.get("root") != str(root.resolve())
            or record.get("experiment") != module.EXPERIMENT_ID or record.get("legacy_runner") != LEGACY):
        raise ValueError("Invalid or relocated runner compatibility registration")
    if set(record["approved_files"]) != set(files(module)):
        raise ValueError("Incomplete runner compatibility registration")
    for name, source in files(module).items():
        expected = record["approved_files"][name]
        if content_identity(source.read_bytes()) != expected:
            raise ValueError(f"Registered code changed: {name}; runner compatibility refused")
        if content_identity((directory / f"{name}.source").read_bytes()) != expected:
            raise ValueError(f"Compatibility source snapshot changed: {name}")
    if content_identity((directory / "legacy_runner.source").read_bytes()) != LEGACY:
        raise ValueError("Legacy runner snapshot changed")
    return record


def common(value):
    return {key: item for key, item in value.items()
            if key not in ("runner", "condition", "network", "seeds")}


def apply(args, value, module):
    record = read_registration(args.exp_root, module)
    if record is None:
        return value
    if common(value) != record["baseline_common"]:
        raise ValueError("Registered protocol parameters/code/reference changed")
    checkpoint = value["condition"]["checkpoint"]
    expected = record["networks"].get(checkpoint)
    if expected is not None and value["network"] != expected:
        raise ValueError("Registered checkpoint model changed")
    return dict(value, runner=dict(LEGACY))


def reviewed_source(module):
    relative = Path(module.__file__).resolve().relative_to(module.REPO_ROOT)
    source = subprocess.check_output(["git", "show", f":{relative}"], cwd=module.REPO_ROOT)
    if content_identity(source) != LEGACY:
        raise ValueError("Git index does not contain the approved legacy runner")
    old = {node.name: ast.dump(node, include_attributes=False)
           for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    new = {node.name: ast.dump(node, include_attributes=False)
           for node in ast.parse(Path(module.__file__).read_bytes()).body if isinstance(node, ast.FunctionDef)}
    if any(old.get(name) != new.get(name) or name not in old for name in STABLE_FUNCTIONS):
        raise ValueError("Sampling/validation functions differ from reviewed legacy runner")
    return source


def validate_existing(args, module):
    rows = {row["condition"]: row for row in module.load_conditions(args.config)}
    protocols, networks, baseline = {}, {}, None
    images = args.exp_root / "images"
    for folder in sorted(images.iterdir()):
        if not folder.is_dir() or folder.name not in rows:
            raise ValueError(f"Unknown image condition: {folder}")
        row = rows[folder.name]
        for directory in sorted(folder.iterdir()):
            if not directory.is_dir():
                raise ValueError(f"Unexpected condition artifact: {directory}")
            if directory.name.startswith("fid-n"):
                continue
            name = directory.name.removesuffix("-staging")
            seeds = None if name == "pool" else name.removeprefix("samples-seeds-")
            if name != "pool" and not name.startswith("samples-seeds-"):
                raise ValueError(f"Unknown image directory: {directory}")
            value = module.protocol(row, args, seeds)
            value = dict(value, runner=dict(LEGACY))
            module.ensure_protocol(directory, value, create=False)
            if not (directory / ".protocol.json").exists():
                raise ValueError(f"Missing protocol: {directory}")
            allowed = range(module.MAX_IMAGES) if seeds is None else module.parse_seeds(seeds)
            module.inventory(directory, allowed)
            protocols[(folder.name, name)] = value
            if baseline is not None and common(value) != baseline:
                raise ValueError("Existing protocols disagree on computation settings")
            baseline = common(value)
            if row["checkpoint"] in networks and networks[row["checkpoint"]] != value["network"]:
                raise ValueError("Existing model identities disagree")
            networks[row["checkpoint"]] = value["network"]
        pool_value = protocols.get((folder.name, "pool"))
        for view in folder.glob("fid-n*"):
            match = re.fullmatch(r"fid-n(5000|25000|50000)-seeds-0-(\d+)", view.name)
            if match is None or pool_value is None or int(match[2]) != int(match[1]) - 1:
                raise ValueError(f"Invalid FID view: {view}")
            count = int(match[1])
            module.ensure_protocol(view, {"pool_protocol_id": module.digest(pool_value), "num_images": count}, create=False)
            actual = module.inventory(view, range(count), complete=True)
            for seed, path in actual.items():
                pooled = folder / "pool" / module.image_relative(seed)
                if not pooled.is_file() or (not os.path.samefile(path, pooled) and path.read_bytes() != pooled.read_bytes()):
                    raise ValueError(f"FID view differs from pool: {path}")
    results = module.load_results(args.exp_root)
    for result in results:
        condition, count = result["condition"], result["num_images"]
        value = protocols.get((condition, "pool"))
        if value is None or result["protocol"] != value:
            raise ValueError("Completed result disagrees with legacy pool")
        module.inventory(images / condition / "pool", range(module.MAX_IMAGES))
        module.inventory(images / condition / f"fid-n{count}-seeds-0-{count - 1}", range(count), complete=True)
    for path in (args.exp_root / "progress").glob("*.json"):
        match = re.fullmatch(r"(.+)-(fid(5000|25000|50000)|samples)\.json", path.name)
        if match is None:
            raise ValueError(f"Unknown progress file: {path}")
        condition = match[1]
        if match[2] == "samples":
            candidates = [value for (name, directory), value in protocols.items()
                          if name == condition and directory.startswith("samples-seeds-")]
        else:
            candidates = [protocols.get((condition, "pool"))]
        progress = json.loads(path.read_text())
        if not any(value and module.digest(value) == progress.get("protocol_id") for value in candidates):
            raise ValueError(f"Progress protocol mismatch: {path}")
        for key in ("generate_sec", "fid_sec", "elapsed_sec"):
            if not math.isfinite(progress[key]) or progress[key] < 0:
                raise ValueError(f"Invalid progress time: {path}")
        if not isinstance(progress["generated_images"], int) or progress["generated_images"] < 0:
            raise ValueError(f"Invalid progress image count: {path}")
    if baseline is None:
        raise ValueError("No legacy image protocols to register")
    return baseline, networks


def register(args, module):
    with (args.exp_root / "logs/.run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = read_registration(args.exp_root, module)
        legacy = ((args.exp_root / DIRECTORY / "legacy_runner.source").read_bytes()
                  if existing else reviewed_source(module))
        baseline, networks = validate_existing(args, module)
        if existing:
            if existing["baseline_common"] != baseline or any(
                    networks.get(key) != item for key, item in existing["networks"].items()):
                raise ValueError("Existing registration disagrees with data")
            print("Runner compatibility already registered; no files changed", flush=True)
            return
        directory = args.exp_root / DIRECTORY
        directory.mkdir(exist_ok=True)
        sources = {name: source.read_bytes() for name, source in files(module).items()}
        for name, data in dict(sources, legacy_runner=legacy).items():
            path = directory / f"{name}.source"
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Refusing to overwrite source snapshot: {path}")
            if not path.exists():
                path.write_bytes(data)
        record = dict(schema=1, root=str(args.exp_root.resolve()), experiment=module.EXPERIMENT_ID,
                      registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      legacy_runner=LEGACY, approved_files={name: content_identity(data) for name, data in sources.items()},
                      baseline_common=baseline, networks=networks)
        module.write_json(directory / "registration.json", record)
        read_registration(args.exp_root, module)
        print(f"Registered runner compatibility: {directory / 'registration.json'}", flush=True)


def record_execution(args, module):
    record = read_registration(args.exp_root, module)
    if record is not None:
        value = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(),
                     compatibility_basis=record["legacy_runner"], actual_files=record["approved_files"],
                     checkpoint_filter=args.checkpoint_filter, condition_filter=args.condition_filter,
                     fid_stage=args.fid_stage, samples_only=args.samples_only)
        module.write_json(args.exp_root / "logs/runner-executions" / f"{time_key()}-{os.getpid()}.json", value)


def time_key():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
