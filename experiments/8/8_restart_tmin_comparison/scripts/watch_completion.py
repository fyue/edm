#!/usr/bin/env python3
"""Watch the CPU-visible experiment state and audit all configured results after GPU work.

This process never starts GPU work or retries the experiment. It records its
state in results/completion.json and reports an incomplete stopped/failed run.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import run_experiment as experiment


def running(root):
    path = root / "logs/.run.lock"
    if not path.exists():
        return False
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def verify_completion(root, rows, results):
    expected = {(row["condition"], count) for row in rows for count in experiment.STAGES}
    actual = {(row["condition"], int(row["num_images"])) for row in results}
    if actual != expected or len(results) != len(expected):
        raise ValueError("Incomplete or duplicate result matrix")
    indexed = {(row["condition"], int(row["num_images"])): row for row in results}
    baseline = json.loads((root / "results/extension_baseline.json").read_text())
    for name, checksum in baseline["result_sha256"].items():
        if hashlib.sha256((root / "results" / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Existing result changed: {name}")
    additional_baseline = json.loads((root / "results/170k_extension_baseline.json").read_text())
    for name, checksum in additional_baseline["file_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Existing result/protocol changed: {name}")
    for row in rows:
        args = argparse.Namespace(exp_root=root, gen_batch=64, fid_batch=64,
                                  fid_ref=experiment.REPO_ROOT / "fid-refs/cifar10-32x32.npz")
        value = experiment.protocol(row, args)
        run_id = experiment.digest(value)
        pool = root / "images" / row["condition"] / "pool"
        experiment.ensure_protocol(pool, value, create=False)
        experiment.ensure_extension(pool, value, create=False)
        images = experiment.inventory(pool, range(50000), complete=True)
        for count in experiment.STAGES:
            result = indexed[row["condition"], count]
            if result["protocol_id"] != run_id or result["seeds"] != f"0-{count - 1}":
                raise ValueError(f"Result protocol/seed mismatch: {row['condition']}, {count}")
            view = pool.parent / f"fid-n{count}-seeds-0-{count - 1}"
            experiment.ensure_protocol(view, {"pool_protocol_id": run_id, "num_images": count}, create=False)
            entries = experiment.inventory(view, range(count), complete=True, verify=False)
            for seed, path in entries.items():
                if not os.path.samefile(images[seed], path) and images[seed].read_bytes() != path.read_bytes():
                    raise ValueError(f"FID view differs from pool: {path}")
    for count in experiment.STAGES:
        for suffix in ("", "_lines"):
            for extension in ("png", "svg"):
                path = root / "results" / f"restart_tmin_fid{count}{suffix}.{extension}"
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Missing result plot: {path}")
        svg = (root / "results" / f"restart_tmin_fid{count}_lines.svg").read_text()
        labels = [experiment.checkpoint_label(checkpoint) for checkpoint in experiment.CHECKPOINTS]
        labels += ["t_min=0.06", "t_min=0.14"]
        if not all(label in svg for label in labels):
            raise ValueError(f"Line plot missing checkpoint/condition labels: FID-{count}")
    return {"completed_evaluations": len(results), "total_pool_images": len(rows) * 50000,
            "preserved_baseline_results": len(baseline["result_sha256"]),
            "preserved_170k_baseline_results": len(additional_baseline["result_protocol_ids"]),
            "line_plot_stages": list(experiment.STAGES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, default=experiment.DEFAULT_ROOT)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--verify-now", action="store_true")
    parser.add_argument("--wait-for-run", action="store_true", help="wait for the main run lock before auditing completion")
    parser.add_argument("--runner-pid", type=int, help="fail if the launching runner exits before acquiring its lock")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be 1..60 seconds")
    os.chdir(experiment.REPO_ROOT)
    root = args.exp_root.resolve()
    rows = experiment.load_conditions(experiment.DEFAULT_ROOT / "config/restart_tmin.tsv")
    report = root / "results/completion.json"
    if args.wait_for_run and not args.verify_now:
        while not running(root):
            if args.runner_pid is not None:
                try:
                    os.kill(args.runner_pid, 0)
                except ProcessLookupError:
                    experiment.write_json(report, {"status": "failed", "error": "Runner exited before acquiring the run lock",
                                                  "expected_evaluations": len(rows) * len(experiment.STAGES)})
                    return 1
            if (root / "STOP_AFTER_CURRENT_CONDITION").exists():
                experiment.write_json(report, {"status": "stopped", "expected_evaluations": len(rows) * len(experiment.STAGES)})
                return 1
            experiment.write_json(report, {"status": "waiting", "completed_evaluations": len(experiment.load_results(root)),
                                          "expected_evaluations": len(rows) * len(experiment.STAGES),
                                          "updated_at_jst": datetime.now(timezone(timedelta(hours=9))).isoformat()})
            time.sleep(args.interval)
    previous = None
    while True:
        results = experiment.load_results(root)
        active = running(root)
        state = {"status": "running" if active else "incomplete",
                 "completed_evaluations": len(results), "expected_evaluations": len(rows) * len(experiment.STAGES),
                 "updated_at_jst": datetime.now(timezone(timedelta(hours=9))).isoformat()}
        if args.verify_now or not active:
            try:
                state.update(verify_completion(root, rows, results))
                state["status"] = "complete"
            except (OSError, ValueError, SyntaxError) as error:
                state["status"] = "stopped" if (root / "STOP_AFTER_CURRENT_CONDITION").exists() else "failed"
                state["error"] = str(error)
            experiment.write_json(report, state)
            print(json.dumps(state, sort_keys=True), flush=True)
            return 0 if state["status"] == "complete" else 1
        experiment.write_json(report, state)
        key = state["status"], len(results)
        if key != previous:
            print(json.dumps(state, sort_keys=True), flush=True)
            previous = key
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
