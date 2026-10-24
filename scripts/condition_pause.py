"""Condition-boundary pause with the shared GPU policy and ETA observer."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pause_eta


REPO_ROOT = Path(__file__).resolve().parent.parent


class StopRequested(Exception):
    """The user requested a clean exit after the current condition."""


class ConditionPause:
    def __init__(self, root, experiment, summary, fid_ref):
        self.root = Path(root)
        self.logs = self.root / 'logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        self.enabled = os.environ.get('PAUSE_ENABLED', '1')
        if self.enabled not in ('0', '1'):
            raise ValueError('PAUSE_ENABLED must be 0 or 1')
        self.enabled = self.enabled == '1'
        self.pause_file = Path(os.environ.get('PAUSE_FILE', '/data/kyou/STOP_RUNNING_PASCAL'))
        if not self.pause_file.is_absolute():
            raise ValueError('PAUSE_FILE must be absolute')
        self.poll = float(os.environ.get('PAUSE_POLL_SEC', '1'))
        self.update = float(os.environ.get('PAUSE_ETA_UPDATE_SEC', '30'))
        if any(not math.isfinite(v) or v <= 0 for v in (self.poll, self.update)):
            raise ValueError('Pause intervals must be positive and finite')
        self.stop_file = self.root / 'STOP_AFTER_CURRENT_CONDITION'
        self.state = self.logs / f'pause-runtime.{os.getpid()}.json'
        self.history = self.logs / 'pause-timings.jsonl'
        self.info = dict(experiment=experiment, seeds='-', run_id=f'{os.getpid()}-{time.time_ns()}',
                         condition='-', checkpoint='-', k='-', num_images=50000,
                         fid_ref=str(fid_ref), log=None)
        self.monitor = None
        self.monitor_log = None
        self.phase('boundary')
        if self.enabled:
            self.monitor_log = (self.logs / 'master.log').open('a')
            try:
                self.monitor = subprocess.Popen([
                    sys.executable, str(REPO_ROOT / 'scripts/pause_eta.py'), 'watch',
                    '--state', str(self.state), '--history', str(self.history),
                    '--pause-file', str(self.pause_file), '--summary', str(summary),
                    '--parent-pid', str(os.getpid()), '--poll-sec', str(self.poll),
                    '--update-sec', str(self.update)], stdout=self.monitor_log, stderr=subprocess.STDOUT)
            except OSError as exc:
                self.event(f'STOP ETA warning: {exc}; condition-boundary pause remains active')

    def event(self, message):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}\n'
        print(line, end='', flush=True)
        for name in ('master.log', 'pause.log'):
            with (self.logs / name).open('a') as handle:
                handle.write(line)

    def phase(self, phase, row=None, seeds=None, log=None):
        if row is not None:
            self.info.update(condition=row['condition'], checkpoint=row['checkpoint'], k=row['restart_k'])
        if seeds is not None:
            self.info['seeds'] = seeds
        if log is not None:
            self.info['log'] = log
        if self.enabled:
            try:
                pause_eta.update_state(SimpleNamespace(state=self.state, history=self.history,
                                                       phase=phase, **self.info))
            except (OSError, ValueError) as exc:
                self.event(f'STOP ETA warning: {exc}; condition-boundary pause remains active')

    def check_exit(self):
        if self.stop_file.exists():
            self.event(f'Found stop file; exiting before next condition: {self.stop_file}')
            raise StopRequested()

    def boundary(self, row, seeds):
        self.phase('boundary', row, seeds)
        while True:
            self.check_exit()
            if self.enabled and self.pause_file.exists():
                self.phase('waiting')
                self.event(f'PAUSE before [{row["condition"]} seeds={seeds}]; waiting for removal: {self.pause_file}')
                while self.pause_file.exists():
                    self.check_exit()
                    time.sleep(self.poll)
            self.check_exit()
            self.phase('gpu_wait')
            self.event(f'RESUME checking configured GPU before [{row["condition"]} seeds={seeds}]')
            env = dict(os.environ, RUN_SCRIPT='/bin/true', EXP_ROOT=str(self.root),
                       LOG_DIR=str(self.logs), PLAN_ONLY='0', SCHEDULE_LABEL=f'{self.info["experiment"]} {row["condition"]} {seeds}')
            # This process only waits; it never launches GPU work. A clean-exit request
            # can terminate the waiter without interrupting an evaluation.
            with subprocess.Popen(['bash', 'scripts/wait_for_gpu_and_run.sh'], cwd=REPO_ROOT, env=env) as waiter:
                try:
                    while waiter.poll() is None:
                        self.check_exit()
                        time.sleep(min(self.poll, 1))
                    if waiter.returncode:
                        raise RuntimeError('GPU policy/wait failed')
                except BaseException:
                    waiter.terminate()
                    waiter.wait()
                    raise
            code = 'import json,os; print(json.dumps({k:os.environ[k] for k in ["GPU_INDEX","EXPECTED_GPU_NAME","CUDA_DEVICE_ORDER","CUDA_VISIBLE_DEVICES"]}))'
            settings = subprocess.check_output(['bash', '-c',
                'set -e; source scripts/load_gpu_config.sh; exec "$1" -c "$2"',
                'gpu-policy', sys.executable, code], cwd=REPO_ROOT, text=True)
            os.environ.update(json.loads(settings))
            self.check_exit()
            if self.enabled and self.pause_file.exists():
                continue
            self.event(f'RESUME GPU free; continuing [{row["condition"]} seeds={seeds}]')
            return

    def close(self, phase='complete'):
        self.phase(phase)
        if self.monitor is not None:
            self.monitor.terminate()
            self.monitor.wait(timeout=10)
        if self.monitor_log is not None:
            self.monitor_log.close()
