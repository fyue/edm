#!/usr/bin/env python3
"""Observe an 8-4 condition and append pause ETA reports to an existing STOP file."""

import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import statistics
import tempfile
import threading
import time


TERMINAL = {"complete", "error", "exit"}
PHASE_NAMES = {
    "boundary": "次条件の開始前", "samples": "固定seedサンプル生成中",
    "generate": "画像生成中", "validate": "画像検証中", "fid": "FID計算中",
    "save": "結果保存中", "saved": "結果保存済み・次条件判定中",
    "waiting": "待機中", "gpu_wait": "再開用GPU確認中",
    "complete": "実験終了・待機不要", "error": "エラー終了", "exit": "runner終了",
}


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False)
    os.replace(temporary, path)


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def append_locked(path, text, existing_only=False, identity=None):
    """Never recreate a deleted STOP, follow symlinks, or write a replaced inode."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK | os.O_NOFOLLOW
    if not existing_only:
        flags |= os.O_CREAT
    try:
        fd = os.open(path, flags, 0o644)
    except FileNotFoundError:
        if existing_only:
            return False
        raise
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Not a regular file: {path}")
        try:
            current = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return False
        key = (opened.st_dev, opened.st_ino)
        if key != (current.st_dev, current.st_ino) or (identity and key != identity):
            return False
        data = text.encode("utf-8")
        while data:
            data = data[os.write(fd, data):]
        return True
    finally:
        os.close(fd)


def update_state(args):
    now = time.monotonic()
    old = read_json(args.state)
    expected_next = {"validate": "fid", "save": "saved"}
    if (expected_next.get(old.get("phase")) == args.phase
            and old.get("run_id") == args.run_id):
        record = {"host": socket.gethostname(), "num_images": old["num_images"],
                  "fid_ref": old["fid_ref"], "phase": old["phase"],
                  "seconds": max(0, now - old["started"])}
        append_locked(args.history, json.dumps(record) + "\n")
    offset = 0
    if args.log and args.log.is_file():
        offset = args.log.stat().st_size
    value = {"phase": args.phase, "started": now, "run_id": args.run_id,
             "condition": args.condition, "checkpoint": args.checkpoint, "k": args.k,
             "num_images": args.num_images, "fid_ref": args.fid_ref,
             "log": str(args.log) if args.log else "", "log_offset": offset}
    atomic_json(args.state, value)


def seconds(value):
    try:
        parts = [int(piece) for piece in value.strip().split(":")]
        if not 1 <= len(parts) <= 3:
            return None
        result = 0
        for piece in parts:
            result = result * 60 + piece
        return result
    except ValueError:
        return None


def progress_remaining(state, now):
    path = Path(state["log"]) if state.get("log") else None
    if path is None or not path.is_file():
        return None
    info = path.stat()
    with path.open("rb") as handle:
        handle.seek(max(state["log_offset"], info.st_size - 65536))
        text = handle.read().decode("utf-8", errors="replace")
    matches = list(re.finditer(r"(\d+)/(\d+)\s*\[([\d:]+)<([^,\]]+)", text))
    if not matches:
        return None
    match = matches[-1]
    completed, total = int(match[1]), int(match[2])
    if completed >= total or not completed:
        return None
    remaining = seconds(match[4])
    if remaining is None:
        elapsed = seconds(match[3])
        remaining = elapsed * (total - completed) / completed if elapsed else None
    if remaining is None:
        return None
    # A stale progress line must not produce a false zero/finished report.
    remaining -= max(0, time.time() - info.st_mtime)
    return remaining if remaining > 0 else None


def median_fid(summary, state):
    if not summary.is_file():
        return None
    values = []
    with summary.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (row.get("num_images") != str(state["num_images"])
                    or row.get("fid_ref") != state["fid_ref"]):
                continue
            try:
                value = float(row["fid_sec"])
                if math.isfinite(value) and value > 0:
                    values.append(value)
            except (KeyError, ValueError):
                pass
    return statistics.median(values[-5:]) if values else None


def overhead_medians(history, state):
    values = {"validate": [], "save": []}
    if history.is_file():
        # History is local to this protocol; host/ref/count prevent mixed estimates.
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if (row["host"] == socket.gethostname() and row["num_images"] == state["num_images"]
                        and row["fid_ref"] == state["fid_ref"]
                        and row["phase"] in values and math.isfinite(row["seconds"])
                        and row["seconds"] >= 0):
                    values[row["phase"]].append(row["seconds"])
            except (ValueError, KeyError, TypeError):
                continue
    return {key: statistics.median(items[-5:]) if items else None for key, items in values.items()}


def estimate(state, summary, history, now=None):
    now = time.monotonic() if now is None else now
    phase = state["phase"]
    if phase in {"waiting", "complete"}:
        return 0, PHASE_NAMES[phase]
    if phase not in {"generate", "validate", "fid", "save"}:
        return None, "推定中" if phase not in TERMINAL else "処理終了"
    elapsed = max(0, now - state["started"])
    fid = median_fid(summary, state)
    overhead = overhead_medians(history, state)
    basis = []
    if phase == "generate":
        remaining = progress_remaining(state, now)
        if remaining is None:
            return None, "生成進捗待ち・再推定中"
        if fid is None:
            return None, f"生成残り約{math.ceil(remaining)}秒／FID所要時間未計測"
        remaining += fid
        basis.append("生成進捗＋直近FID中央値")
    elif phase == "fid":
        progress = progress_remaining(state, now)
        if fid is not None:
            remaining = max(fid - elapsed, progress or 0)
            if remaining <= 0:
                return None, "見積もり超過・処理継続中"
            basis.append("直近FID中央値・現在の進捗")
        elif progress is not None:
            remaining = progress
            basis.append("FID進捗（FID後処理未計測）")
        else:
            return None, "FID所要時間未計測・推定中"
    else:
        measured = overhead[phase]
        if measured is None:
            return None, "付帯処理時間未計測・推定中"
        remaining = measured - elapsed
        if remaining <= 0:
            return None, "見積もり超過・処理継続中"
        if phase == "validate":
            if fid is None:
                return None, "FID所要時間未計測・推定中"
            remaining += fid
        basis.append("付帯処理・FIDの直近実測中央値")
    later = {"generate": ["validate", "save"], "validate": ["save"],
             "fid": ["save"], "save": []}[phase]
    missing = False
    for key in later:
        if overhead[key] is None:
            missing = True
        else:
            remaining += overhead[key]
    if missing:
        basis.append("検証・保存時間は一部未計測")
    return remaining, "／".join(basis)


def report(state, summary, history, parent_pid):
    remaining, basis = estimate(state, summary, history)
    timestamp = datetime.now().astimezone()
    if state["phase"] in {"error", "exit"}:
        duration, target = "対象外", "-"
    elif remaining is None:
        duration, target = "推定中", "未定"
    elif remaining == 0:
        duration, target = "0秒", timestamp.isoformat(timespec="seconds")
    else:
        duration = f"約{math.ceil(remaining / 60)}分" if remaining >= 60 else f"約{math.ceil(remaining)}秒"
        target = (timestamp + timedelta(seconds=remaining)).isoformat(timespec="seconds")
    return (
        f"\n[{timestamp.isoformat(timespec='seconds')}] experiment=8-4 host={socket.gethostname()} "
        f"pid={parent_pid} run={state['run_id']} checkpoint={state['checkpoint']} "
        f"K={state['k']} FID={state['num_images']}\n"
        f"状態={PHASE_NAMES[state['phase']]} 待機開始まで={duration} 待機開始見込み={target} 根拠={basis}\n"
    )


def watch(args):
    stopping = threading.Event()

    def stop(_signum, _frame):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_key, last_update, last_warning = None, 0, None
    while True:
        state = read_json(args.state)
        orphaned = os.getppid() != args.parent_pid
        if orphaned and state and state["phase"] not in TERMINAL:
            state["phase"] = "exit"
        try:
            info = os.stat(args.pause_file, follow_symlinks=False)
            identity = (info.st_dev, info.st_ino)
            key = (identity, state.get("phase"), state.get("started"))
            if state and (key != last_key or time.monotonic() - last_update >= args.update_sec or stopping.is_set() or orphaned):
                if append_locked(args.pause_file, report(state, args.summary, args.history, args.parent_pid),
                                 existing_only=True, identity=identity):
                    last_key, last_update = key, time.monotonic()
                    last_warning = None
        except FileNotFoundError:
            last_key = None
        except (OSError, ValueError, KeyError) as exc:
            warning = f"STOP ETA warning: {exc}"
            if warning != last_warning:
                print(warning, flush=True)
                last_warning = warning
        if stopping.is_set() or orphaned:
            return
        stopping.wait(args.poll_sec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    writer = subparsers.add_parser("state")
    monitor = subparsers.add_parser("watch")
    for command in [writer, monitor]:
        command.add_argument("--state", type=Path, required=True)
        command.add_argument("--history", type=Path, required=True)
    writer.add_argument("--phase", choices=list(PHASE_NAMES), required=True)
    writer.add_argument("--run-id", required=True)
    writer.add_argument("--condition", default="-")
    writer.add_argument("--checkpoint", default="-")
    writer.add_argument("--k", default="-")
    writer.add_argument("--num-images", type=int, required=True)
    writer.add_argument("--fid-ref", required=True)
    writer.add_argument("--log", type=Path)
    monitor.add_argument("--pause-file", type=Path, required=True)
    monitor.add_argument("--summary", type=Path, required=True)
    monitor.add_argument("--parent-pid", type=int, required=True)
    monitor.add_argument("--poll-sec", type=float, required=True)
    monitor.add_argument("--update-sec", type=float, required=True)
    args = parser.parse_args()
    if args.command == "state":
        update_state(args)
    else:
        for value in [args.poll_sec, args.update_sec]:
            if not math.isfinite(value) or value <= 0:
                parser.error("Monitor intervals must be positive and finite")
        watch(args)


if __name__ == "__main__":
    main()
