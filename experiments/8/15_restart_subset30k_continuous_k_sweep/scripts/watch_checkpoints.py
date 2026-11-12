#!/usr/bin/env python3
"""Watch 0-6 snapshots and evaluate experiment 8-15, together or one mode at a time."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
MODES = ('main18', 'nfe55')
sys.path.insert(0, str(REPO / 'scripts'))
from condition_pause import ConditionPause, StopRequested


def engine(mode):
    name = 'experiment815_' + mode
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/run_experiment.py')
    module = importlib.util.module_from_spec(spec)
    old = os.environ.get('FIXED_MODE')
    os.environ['FIXED_MODE'] = mode
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop('FIXED_MODE', None)
        else:
            os.environ['FIXED_MODE'] = old
    return module


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another 8-15 runner holds {path}') from error
        yield


def signature(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def cpu_validate(path):
    # Training saves CPU modules. Hide CUDA as an additional guard during loading.
    code = '''import pickle, sys
import torch
with open(sys.argv[1], 'rb') as handle:
    data = pickle.load(handle)
net = data.get('ema') if isinstance(data, dict) else None
if not isinstance(net, torch.nn.Module):
    raise ValueError('Snapshot has no EMA torch module')
if (net.img_resolution, net.img_channels, net.label_dim) != (32, 3, 10):
    raise ValueError('Expected conditional CIFAR-10 model')
if any(t.device.type != 'cpu' for t in list(net.parameters()) + list(net.buffers())):
    raise ValueError('Snapshot tensors must be on CPU')
print('CPU snapshot validation OK', flush=True)
'''
    subprocess.run([sys.executable, '-c', code, str(path)], cwd=REPO,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)


class SnapshotProbe:
    def __init__(self, stable_sec=30, validate=cpu_validate, clock=time.monotonic):
        self.stable_sec, self.validate, self.clock = stable_sec, validate, clock
        self.observed = {}
        self.validated = {}

    def ready(self, path):
        current = signature(path)
        if current is None or current[0] == 0:
            self.observed.pop(path, None)
            self.validated.pop(path, None)
            return False
        now = self.clock()
        prior, since = self.observed.get(path, (None, now))
        if prior != current:
            self.observed[path] = current, now
            self.validated.pop(path, None)
            return False
        if now - since < self.stable_sec:
            return False
        if self.validated.get(path) == current:
            return True
        try:
            self.validate(path)
        except Exception:
            if signature(path) != current:
                self.observed.pop(path, None)
                return False
            raise
        if signature(path) != current:
            self.observed.pop(path, None)
            return False
        self.validated[path] = current
        return True


class ExperimentPause(ConditionPause):
    def __init__(self, *args, overall_root, **kwargs):
        self.overall_root = overall_root
        super().__init__(*args, **kwargs)

    def check_exit(self):
        super().check_exit()
        if (self.overall_root / 'STOP_AFTER_CURRENT_CONDITION').exists():
            self.event('Experiment-wide STOP_AFTER_CURRENT_CONDITION; exiting')
            raise StopRequested()


def sample_complete(module, row, args):
    seeds = module.parse_seeds(args.sample_seeds)
    directory = args.exp_root / 'images' / row['condition'] / f'samples-seeds-{args.sample_seeds}'
    module.ensure_protocol(directory, module.protocol(row, args, args.sample_seeds), create=False)
    images = module.inventory(directory, seeds)
    return set(images) == set(seeds)


class Evaluation:
    def __init__(self, module, args, rows):
        self.module, self.args, self.rows = module, args, rows
        self.results = module.load_results(args.exp_root)
        self.pause = None

    def completed(self, rows):
        recorded = {r['condition']: r for r in self.results}
        for row in rows:
            if row['condition'] in recorded:
                if not Path(row['network']).is_file():
                    raise ValueError(f"Recorded result model missing: {row['network']}")
                value = self.module.protocol(row, self.args)
                if recorded[row['condition']]['protocol_id'] != self.module.digest(value):
                    raise ValueError('Recorded result protocol changed')
                pool = self.args.exp_root / 'images' / row['condition'] / 'pool'
                view = pool.parent / 'fid-n50000-seeds-0-49999'
                self.module.ensure_protocol(pool, value, create=False)
                self.module.inventory(pool, range(50000), complete=True)
                self.module.ensure_protocol(view, {'pool_protocol_id': self.module.digest(value),
                                                   'num_images': 50000}, create=False)
                self.module.inventory(view, range(50000), complete=True)
                print(f"RESUME {self.module.MODE} {row['condition']}: verified; skipping", flush=True)
        return all(row['condition'] in recorded for row in rows)

    def samples(self, rows):
        self.args.samples_only = True
        for row in rows:
            if not sample_complete(self.module, row, self.args):
                self.module.run_condition(row, 0, self.args, self.results, self.pause)

    def evaluate(self, rows):
        self.args.samples_only = False
        recorded = {r['condition'] for r in self.results}
        for row in rows:
            if row['condition'] not in recorded:
                self.module.run_condition(row, 50000, self.args, self.results, self.pause)


def run_watch(evaluations, checkpoints, probe, poll, wait=time.sleep, samples_only=False):
    finished = set()
    while True:
        pending = []
        for checkpoint in checkpoints:
            if checkpoint in finished:
                continue
            groups = [(evaluation, [r for r in evaluation.rows if r['checkpoint'] == checkpoint])
                      for evaluation in evaluations]
            path = Path(groups[0][1][0]['network'])
            # Validate completed work before considering STOP/GPU waits.
            if not samples_only:
                if all([evaluation.completed(rows) for evaluation, rows in groups]):
                    finished.add(checkpoint)
                    continue
            pending.append((checkpoint, path, groups))
        if not pending:
            for evaluation in evaluations:
                if not samples_only:
                    expected = {row['condition'] for row in evaluation.rows}
                    actual = {row['condition'] for row in evaluation.results}
                    if not expected.issubset(actual):
                        raise ValueError('Final aggregation has missing completed conditions')
                evaluation.module.aggregate(evaluation.args.exp_root, evaluation.results)
            print(f'COMPLETE {len(checkpoints) * len(evaluations) * 9} conditions', flush=True)
            return
        for evaluation in evaluations:
            evaluation.pause.check_exit()
        progressed = False
        for checkpoint, path, groups in pending:
            if not probe.ready(path):
                print(f'WAIT checkpoint={checkpoint} missing/writing/stabilizing: {path}', flush=True)
                continue
            print(f'READY checkpoint={checkpoint}; CPU validation passed', flush=True)
            for evaluation, rows in groups:
                evaluation.args.snapshot_signature = probe.validated[path]
                evaluation.samples(rows)
            if not samples_only:
                for evaluation, rows in groups:
                    evaluation.evaluate(rows)
            finished.add(checkpoint)
            progressed = True
            print(f'PROGRESS checkpoints={len(finished)}/{len(checkpoints)}', flush=True)
        if progressed:
            continue
        for evaluation in evaluations:
            evaluation.pause.info.update(condition='checkpoint-wait', checkpoint=','.join(
                checkpoint for checkpoint, _, _ in pending), k='-')
            evaluation.pause.phase('waiting')
        # Short sleeps allow explicit exit requests during checkpoint waits.
        remaining = poll
        while remaining > 0:
            for evaluation in evaluations:
                evaluation.pause.check_exit()
            interval = min(1, remaining)
            wait(interval)
            remaining -= interval


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all',) + MODES, default='all')
    parser.add_argument('--checkpoint-filter', default='all')
    parser.add_argument('--checkpoint-poll-sec', type=float,
                        default=float(os.environ.get('CHECKPOINT_POLL_SEC', '300')))
    parser.add_argument('--exp-root', type=Path, default=ROOT,
                        help='Overall experiment root; mode outputs use runs/<mode>')
    parser.add_argument('--fid-ref', type=Path, default=Path('fid-refs/cifar10-32x32.npz'))
    parser.add_argument('--torchrun', default=str(Path(sys.executable).parent / 'torchrun'))
    parser.add_argument('--pause-file', type=Path,
                        default=Path(os.environ.get('PAUSE_FILE', '/data/kyou/STOP_RUNNING_PASCAL')))
    parser.add_argument('--pause-poll-sec', type=float, default=float(os.environ.get('PAUSE_POLL_SEC', '1')))
    parser.add_argument('--samples-only', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    for name in ('plan-only', 'verify-only', 'aggregate-only'):
        modes.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    args.plan_only = args.plan_only or os.environ.get('PLAN_ONLY') == '1'
    for value in (args.checkpoint_poll_sec, args.pause_poll_sec):
        if not math.isfinite(value) or value <= 0:
            parser.error('poll intervals must be positive and finite')
    if not args.pause_file.is_absolute():
        parser.error('--pause-file must be absolute')
    args.exp_root = args.exp_root.resolve()
    args.fid_ref = args.fid_ref.resolve()
    return args


def main():
    os.chdir(REPO)
    options = arguments()
    modes = MODES if options.mode == 'all' else (options.mode,)
    evaluations = []
    for mode in modes:
        module = engine(mode)
        rows = module.select_conditions(module.load_conditions(ROOT / 'config' / f'restart_{mode}.tsv'),
                                        options.checkpoint_filter, '')
        args = SimpleNamespace(exp_root=options.exp_root / 'runs' / mode,
                               fid_ref=options.fid_ref, gen_batch=64, fid_batch=64,
                               torchrun=options.torchrun, gpu_name='unknown', sample_seeds='0-1',
                               samples_only=False, pause=None)
        evaluations.append(Evaluation(module, args, rows))
    checkpoints = sorted({r['checkpoint'] for r in evaluations[0].rows}, key=lambda s: int(s[:-1]))
    if options.plan_only:
        for evaluation in evaluations:
            for row in evaluation.rows:
                print(f"PLAN {evaluation.module.MODE} {row['condition']} main={row['main_steps']} "
                      f"NFE={row['nfe']} network={row['network']} present={Path(row['network']).exists()}")
        conditions = len(checkpoints) * len(modes) * 9
        print(f'PLAN conditions={conditions}; FID-50k evaluations={0 if options.samples_only else conditions}; '
              f'max evaluation images={0 if options.samples_only else conditions * 50000}; '
              f'smoke images={conditions * 2}; checkpoint poll={options.checkpoint_poll_sec}s; stable=30s')
        return
    if options.verify_only:
        for evaluation in evaluations:
            for row in evaluation.rows:
                if any(r['condition'] == row['condition'] for r in evaluation.results):
                    evaluation.completed([row])
                if Path(row['network']).exists():
                    sample_complete(evaluation.module, row, evaluation.args)
        print('VERIFY completed results and existing smoke images OK', flush=True)
        return
    with ExitStack() as stack:
        # Mode locks are shared by single-mode and combined runners, held even while waiting.
        if options.mode == 'all':
            stack.enter_context(lock(options.exp_root / 'logs/.sequence.lock'))
        for evaluation in evaluations:
            stack.enter_context(lock(evaluation.args.exp_root / 'logs/.run.lock'))
            evaluation.results = evaluation.module.load_results(evaluation.args.exp_root)
        if options.aggregate_only:
            for evaluation in evaluations:
                evaluation.module.aggregate(evaluation.args.exp_root, evaluation.results)
            return
        for evaluation in evaluations:
            evaluation.module.identity(options.fid_ref)
        os.environ['PAUSE_FILE'] = str(options.pause_file)
        os.environ['PAUSE_POLL_SEC'] = str(options.pause_poll_sec)
        os.environ.setdefault('DNNLIB_CACHE_DIR', str(REPO / '.cache/dnnlib'))
        terminal = 'error'
        try:
            for evaluation in evaluations:
                evaluation.pause = ExperimentPause(
                    evaluation.args.exp_root, evaluation.module.EXPERIMENT_ID,
                    evaluation.args.exp_root / 'results/restart_fid_summary.tsv', options.fid_ref,
                    overall_root=options.exp_root)
            run_watch(evaluations, checkpoints, SnapshotProbe(), options.checkpoint_poll_sec,
                      samples_only=options.samples_only)
            terminal = 'complete'
        except StopRequested:
            terminal = 'exit'
            print('STOP: clean exit before next condition', flush=True)
        finally:
            for evaluation in evaluations:
                if evaluation.pause is not None:
                    evaluation.pause.close(terminal)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
