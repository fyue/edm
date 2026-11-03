#!/usr/bin/env python3
"""After one nfe55 runner exits cleanly, resume only 20k/40k/60k on this host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = (path / 'cmdline').read_bytes().split(b'\0')
        env = dict(entry.split(b'=', 1) for entry in (path / 'environ').read_bytes().split(b'\0') if b'=' in entry)
    except FileNotFoundError:
        return None
    if not any(x.endswith(b'14_restart_subset30k_k_sweep/scripts/run_experiment.py') for x in argv):
        raise RuntimeError('The PID does not belong to the expected experiment runner')
    if env.get(b'FIXED_MODE') != b'nfe55':
        raise RuntimeError('The PID is not an nfe55 runner')
    return stat[19]  # Linux /proc stat field 22, process start ticks.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'runs/nfe55'
    logs = output / 'logs'
    stop = output / 'STOP_AFTER_CURRENT_CONDITION'
    state_path = logs / f'pause-runtime.{args.wait_pid}.json'
    identity = process_identity(args.wait_pid)
    command = ['bash', str(ROOT / 'scripts/run.sh'), '--mode', 'nfe55',
               '--checkpoint-filter', '20k,40k,60k', '--condition-filter', '',
               '--fid-stage', 'all', '--exp-root', str(output),
               '--config', str(ROOT / 'config/restart_nfe55.tsv'), '--resume']
    print(json.dumps({'wait_pid': args.wait_pid, 'start_ticks': identity,
                      'stop_file': str(stop), 'command': command}), flush=True)
    if args.plan_only:
        return
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / '.subset-schedule.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if identity is not None and not stop.exists():
            raise RuntimeError('Current runner has no condition-boundary exit request')
        print('WAIT: keeping the current exit request until the runner finishes', flush=True)
        while identity is not None:
            current = process_identity(args.wait_pid)
            if current is None:
                break
            if current != identity:
                raise RuntimeError('PID was reused; refusing automatic restart')
            time.sleep(5)
        state = json.loads(state_path.read_text())
        if state.get('phase') not in ('exit', 'complete') or state.get('experiment') != '8-14-nfe55':
            raise RuntimeError('Previous runner did not exit cleanly; leaving STOP in place')
        # Do not remove STOP if another runner has already acquired this output.
        with (logs / '.run.lock').open('a') as run_lock:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stop.unlink(missing_ok=True)
        env = dict(os.environ, PYTHON=args.python, PLAN_ONLY='0', VERIFY_ONLY='0',
                   AGGREGATE_ONLY='0', SAMPLES_ONLY='0')
        print('START: nfe55, checkpoints=20k/40k/60k, K=0..8, FID through 50k', flush=True)
        # run.sh and ConditionPause retain the local GPU policy and shared GPU waiter.
        result = subprocess.run(command, cwd=REPO, env=env)
        print(f'FINISHED: runner exit status={result.returncode}', flush=True)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
