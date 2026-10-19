#!/usr/bin/env python3
"""List Codex app-server session locks and stop stale owners safely.

Examples:
    scripts/manage_codex_sessions.py
    scripts/manage_codex_sessions.py --stop-stale
    scripts/manage_codex_sessions.py --stop-thread 'GPU1実行環境を準備'

The list operation is read-only. Stop operations use SIGTERM, require interactive
confirmation by default, and never escalate to SIGKILL.
"""

import argparse
import datetime as dt
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EXTENSION_VERSION_RE = re.compile(r"/openai\.chatgpt-([^/]+)/")


@dataclass(frozen=True)
class ThreadInfo:
    thread_id: str
    name: str
    updated_at: str


@dataclass(frozen=True)
class AppServer:
    pid: int
    ppid: int
    start_ticks: int
    started_at: float
    executable: str
    extension_version: str
    threads: Tuple[ThreadInfo, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Codex app-server と保持中のセッションロックを一覧表示し、"
            "古い重複プロセスを安全に停止します。"
        )
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--stop-stale",
        action="store_true",
        help=(
            "最新の app-server を残し、--min-age-minutes 以上経過した"
            "重複 app-server を SIGTERM で停止する"
        ),
    )
    action.add_argument(
        "--stop-thread",
        metavar="NAME_OR_ID",
        help="完全一致するスレッド名またはIDのロック所有プロセスを停止する",
    )
    parser.add_argument(
        "--min-age-minutes",
        type=int,
        default=30,
        help="--stop-stale の最小経過時間（既定: 30分）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="停止前の対話確認を省略する",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="SIGTERM 後に終了を待つ秒数（既定: 5秒）",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")),
        help="Codexホームディレクトリ（既定: CODEX_HOME または ~/.codex）",
    )
    args = parser.parse_args(argv)
    if args.min_age_minutes < 0:
        parser.error("--min-age-minutes must be zero or greater")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be zero or greater")
    if args.yes and not (args.stop_stale or args.stop_thread):
        parser.error("--yes requires --stop-stale or --stop-thread")
    return args


def read_boot_time() -> float:
    with Path("/proc/stat").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("btime "):
                return float(line.split()[1])
    raise RuntimeError("Could not read system boot time from /proc/stat")


def read_process_stat(pid: int) -> Tuple[int, int]:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing_paren = text.rfind(")")
    if closing_paren < 0:
        raise ValueError(f"Malformed /proc/{pid}/stat")
    fields = text[closing_paren + 2 :].split()
    # fields[0] is field 3 (state); ppid and starttime are fields 4 and 22.
    return int(fields[1]), int(fields[19])


def read_cmdline(pid: int) -> List[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def is_codex_app_server(cmdline: Sequence[str]) -> bool:
    if not cmdline or "app-server" not in cmdline[1:]:
        return False
    executable = cmdline[0]
    return Path(executable).name == "codex" and EXTENSION_VERSION_RE.search(executable) is not None


def load_thread_index(codex_home: Path) -> Dict[str, ThreadInfo]:
    index_path = codex_home / "session_index.jsonl"
    threads: Dict[str, ThreadInfo] = {}
    try:
        stream = index_path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return threads
    with stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
                thread_id = record["id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                print(
                    f"警告: {index_path}:{line_number} を解釈できません",
                    file=sys.stderr,
                )
                continue
            threads[thread_id] = ThreadInfo(
                thread_id=thread_id,
                name=str(record.get("thread_name", "(名前なし)")),
                updated_at=str(record.get("updated_at", "-")),
            )
    return threads


def writer_locked_inodes(pid: int) -> set:
    inodes = set()
    try:
        stream = Path("/proc/locks").open("r", encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return inodes
    with stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 6:
                continue
            if fields[1] != "FLOCK" or fields[3] != "WRITE":
                continue
            try:
                owner_pid = int(fields[4])
                inode = int(fields[5].rsplit(":", maxsplit=1)[1])
            except (ValueError, IndexError):
                continue
            if owner_pid == pid:
                inodes.add(inode)
    return inodes


def process_lock_ids(pid: int, lock_dir: Path) -> List[str]:
    lock_ids = set()
    locked_inodes = writer_locked_inodes(pid)
    if not locked_inodes:
        return []
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        descriptors = list(fd_dir.iterdir())
    except (FileNotFoundError, PermissionError):
        return []
    for descriptor in descriptors:
        try:
            target = Path(os.readlink(descriptor))
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if target.parent != lock_dir or target.suffix != ".lock":
            continue
        if target.name.startswith("."):
            continue
        try:
            inode = target.stat().st_ino
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if inode not in locked_inodes:
            continue
        lock_ids.add(target.stem)
    return sorted(lock_ids)


def find_app_servers(codex_home: Path) -> List[AppServer]:
    boot_time = read_boot_time()
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    thread_index = load_thread_index(codex_home)
    lock_dir = codex_home / "thread-writer-locks"
    servers: List[AppServer] = []

    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        try:
            if proc_dir.stat().st_uid != os.getuid():
                continue
            cmdline = read_cmdline(pid)
            if not is_codex_app_server(cmdline):
                continue
            ppid, start_ticks = read_process_stat(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue

        executable = cmdline[0]
        version_match = EXTENSION_VERSION_RE.search(executable)
        version = version_match.group(1) if version_match else "unknown"
        threads = []
        for thread_id in process_lock_ids(pid, lock_dir):
            threads.append(
                thread_index.get(
                    thread_id,
                    ThreadInfo(thread_id=thread_id, name="(indexに記録なし)", updated_at="-"),
                )
            )
        servers.append(
            AppServer(
                pid=pid,
                ppid=ppid,
                start_ticks=start_ticks,
                started_at=boot_time + start_ticks / clock_ticks,
                executable=executable,
                extension_version=version,
                threads=tuple(threads),
            )
        )
    return sorted(servers, key=lambda server: (server.started_at, server.pid))


def format_duration(seconds: float) -> str:
    total_minutes = max(0, int(seconds // 60))
    days, remainder = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    if days:
        return f"{days}日{hours}時間"
    if hours:
        return f"{hours}時間{minutes}分"
    return f"{minutes}分"


def print_servers(servers: Sequence[AppServer], stale_pids: Iterable[int]) -> None:
    stale_pid_set = set(stale_pids)
    newest_pid = servers[-1].pid if servers else None
    now = time.time()
    if not servers:
        print("Codex app-server は見つかりませんでした。")
        return

    for server in servers:
        if server.pid == newest_pid:
            status = "最新（保護）"
        elif server.pid in stale_pid_set:
            status = "停止候補"
        else:
            status = "重複（年齢条件外）"
        started = dt.datetime.fromtimestamp(server.started_at).astimezone()
        print(
            f"PID {server.pid}  {status}  version={server.extension_version}  "
            f"age={format_duration(now - server.started_at)}  "
            f"started={started:%Y-%m-%d %H:%M:%S %Z}"
        )
        if server.threads:
            for thread in server.threads:
                print(
                    f"  - {thread.thread_id}  {thread.name}  "
                    f"index_updated={thread.updated_at}"
                )
        else:
            print("  - 保持中のスレッドロックなし")


def stale_servers(servers: Sequence[AppServer], min_age_minutes: int) -> List[AppServer]:
    if len(servers) < 2:
        return []
    newest_pid = servers[-1].pid
    minimum_age_seconds = min_age_minutes * 60
    now = time.time()
    return [
        server
        for server in servers
        if server.pid != newest_pid and now - server.started_at >= minimum_age_seconds
    ]


def servers_for_thread(servers: Sequence[AppServer], query: str) -> List[AppServer]:
    id_matches = [
        server
        for server in servers
        if any(thread.thread_id == query for thread in server.threads)
    ]
    if id_matches:
        return id_matches

    matching_ids = {
        thread.thread_id
        for server in servers
        for thread in server.threads
        if thread.name == query
    }
    if len(matching_ids) > 1:
        print(
            f"同名のスレッドが {len(matching_ids)} 件あります。IDを指定してください:",
            file=sys.stderr,
        )
        for thread_id in sorted(matching_ids):
            print(f"  {thread_id}", file=sys.stderr)
        return []
    return [
        server
        for server in servers
        if any(thread.thread_id in matching_ids for thread in server.threads)
    ]


def confirm_stop(servers: Sequence[AppServer]) -> bool:
    if not sys.stdin.isatty():
        print("対話確認できません。停止する場合は --yes を追加してください。", file=sys.stderr)
        return False
    pids = ", ".join(str(server.pid) for server in servers)
    response = input(f"PID {pids} に SIGTERM を送信します。続けますか？ [y/N] ")
    return response.strip().lower() in {"y", "yes"}


def same_process(server: AppServer) -> bool:
    try:
        cmdline = read_cmdline(server.pid)
        _, start_ticks = read_process_stat(server.pid)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return False
    return start_ticks == server.start_ticks and is_codex_app_server(cmdline)


def stop_servers(servers: Sequence[AppServer], timeout_seconds: float) -> bool:
    pending = []
    for server in servers:
        if not same_process(server):
            print(f"PID {server.pid}: 既に終了したか別プロセスになったためスキップ")
            continue
        try:
            os.kill(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"PID {server.pid}: 既に終了しています")
            continue
        except PermissionError as error:
            print(f"PID {server.pid}: 停止できません: {error}", file=sys.stderr)
            pending.append(server)
            continue
        print(f"PID {server.pid}: SIGTERMを送信しました")
        pending.append(server)

    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        pending = [server for server in pending if same_process(server)]
        if pending:
            time.sleep(0.1)

    remaining = [server for server in pending if same_process(server)]
    if remaining:
        print(
            "時間内に終了しなかったPID: "
            + ", ".join(str(server.pid) for server in remaining),
            file=sys.stderr,
        )
        print("強制終了は行っていません。", file=sys.stderr)
        return False
    print("対象プロセスはすべて終了しました。")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    codex_home = args.codex_home.expanduser().resolve()
    servers = find_app_servers(codex_home)
    stale = stale_servers(servers, args.min_age_minutes)
    print_servers(servers, (server.pid for server in stale))

    targets: List[AppServer] = []
    if args.stop_stale:
        targets = stale
        if not targets:
            print(
                f"{args.min_age_minutes}分以上経過した古い重複 app-server はありません。"
            )
            return 0
    elif args.stop_thread:
        targets = servers_for_thread(servers, args.stop_thread)
        if not targets:
            print(
                f"ロック中のスレッドが見つかりません: {args.stop_thread}",
                file=sys.stderr,
            )
            return 1
        print(
            "注意: スレッド単体ではなく、そのロックを保持する app-server を停止します。"
        )

    if not targets:
        print("一覧表示のみです。プロセスは停止していません。")
        return 0
    if not args.yes and not confirm_stop(targets):
        print("停止を中止しました。")
        return 1
    return 0 if stop_servers(targets, args.timeout_seconds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
