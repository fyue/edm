#!/usr/bin/env python
"""Experiment 7-1: FM checkpointの保存・再読込と学習健全性を検証する。

`run_fm_linear_baseline.sh` の smoke profile から呼ばれ、次を確認する。

- `training-state-*.pt` を再読込でき、netの全パラメータが有限である。
- `network-snapshot-*.pkl` を再読込でき、EMA netがvelocity checkpointとして
  正しい属性 (prediction_type, path, 時間範囲) を持つ。
- 再読込したnetでFMLossのforward/backwardが通り、lossとgradientが有限である。
- `stats.jsonl` に記録されたloss・resource値が有限である。

使い方:
    python experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/verify_fm_checkpoint.py \\
        --run-dir=training-runs/cifar10-cond-ddpmpp-fm-linear --kimg=50
"""

import argparse
import json
import math
import os
import pickle
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import dnnlib                                           # noqa: E402
from training.loss import FMLoss                        # noqa: E402

#----------------------------------------------------------------------------

def check(condition, message):
    print(f'{"OK  " if condition else "FAIL"} {message}')
    return bool(condition)

#----------------------------------------------------------------------------

def verify_training_state(path):
    """training-state-*.pt を再読込し、netパラメータの有限性を確認する。"""
    ok = True
    data = torch.load(path, map_location=torch.device('cpu'))
    ok &= check('net' in data and 'optimizer_state' in data,
                f'training state has net and optimizer_state: {os.path.basename(path)}')
    net = data['net']
    num_params = sum(p.numel() for p in net.parameters())
    finite = all(torch.isfinite(p).all() for p in net.parameters())
    ok &= check(finite, f'all {num_params} training-state parameters are finite')
    return ok, net

#----------------------------------------------------------------------------

def verify_snapshot(path):
    """network-snapshot-*.pkl を再読込し、velocity checkpointの規約を確認する。"""
    ok = True
    with open(path, 'rb') as f:
        data = pickle.load(f)
    ema = data['ema']
    ok &= check(getattr(ema, 'prediction_type', None) == 'velocity',
                f"snapshot ema has prediction_type='velocity' (got {getattr(ema, 'prediction_type', None)!r})")
    ok &= check(getattr(ema, 'path', None) == 'linear',
                f"snapshot ema has path='linear' (got {getattr(ema, 'path', None)!r})")
    ok &= check((getattr(ema, 't_min', None), getattr(ema, 't_max', None)) == (0, 1),
                f"snapshot ema time range is [0,1] (got [{getattr(ema, 't_min', None)}, {getattr(ema, 't_max', None)}])")
    ok &= check(getattr(ema, 'M', None) == 1000,
                f'snapshot ema uses M=1000 so that c_noise = 999 * t (got {getattr(ema, "M", None)})')
    ok &= check(not hasattr(ema, 'sigma_min') and not hasattr(ema, 'sigma_max'),
                'snapshot ema exposes no EDM sigma range (EDM samplers must reject it)')
    ok &= check(isinstance(data.get('loss_fn'), FMLoss),
                f'snapshot loss_fn is FMLoss (got {type(data.get("loss_fn")).__name__})')
    ok &= check(getattr(data.get('loss_fn'), 'path', None) == 'linear',
                "snapshot loss_fn uses path='linear'")
    ok &= check(all(torch.isfinite(p).all() for p in ema.parameters()),
                'all EMA parameters are finite')
    return ok, ema, data

#----------------------------------------------------------------------------

def verify_forward_backward(net, dataset_kwargs, batch_size, device):
    """再読込したnetでFMLossのforward/backwardを1回行い、有限性を確認する。"""
    ok = True
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    images = []
    labels = []
    for index in range(batch_size):
        image, label = dataset_obj[index]
        images.append(torch.as_tensor(image))
        labels.append(torch.as_tensor(label))
    images = torch.stack(images).to(device).to(torch.float32) / 127.5 - 1
    labels = torch.stack(labels).to(device).to(torch.float32)

    net = net.to(device).train().requires_grad_(True)
    net.zero_grad(set_to_none=True)
    loss = FMLoss(path='linear')(net=net, images=images, labels=labels)
    ok &= check(loss.shape == images.shape, f'loss is per-pixel with shape {tuple(loss.shape)}')
    ok &= check(bool(torch.isfinite(loss).all()), f'loss is finite (mean={float(loss.mean()):.6g})')
    loss.sum().mul(1 / batch_size).backward()

    grads = [p.grad for p in net.parameters() if p.grad is not None]
    ok &= check(len(grads) > 0, f'{len(grads)} parameter tensors received gradients')
    ok &= check(all(torch.isfinite(g).all() for g in grads), 'all gradients are finite')
    grad_norm = torch.sqrt(sum((g.double() ** 2).sum() for g in grads))
    ok &= check(bool(torch.isfinite(grad_norm)), f'global gradient norm is finite ({float(grad_norm):.6g})')
    net.zero_grad(set_to_none=True)
    return ok

#----------------------------------------------------------------------------

def verify_stats(path):
    """stats.jsonl のloss・resource値が有限であることを確認する。"""
    ok = True
    if not os.path.isfile(path):
        return check(False, f'missing stats file: {path}')

    bad = []
    rows = 0
    with open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            for key, value in row.items():
                if not key.startswith(('Loss/', 'Resources/', 'Timing/')):
                    continue
                stats = value if isinstance(value, dict) else {'value': value}
                for stat_key, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)) and not math.isfinite(stat_value):
                        bad.append(f'{key}.{stat_key}={stat_value}')
    ok &= check(rows > 0, f'stats.jsonl has {rows} tick records')
    ok &= check(not bad, 'all recorded loss/resource/timing values are finite'
                + ('' if not bad else f' (bad: {", ".join(bad[:5])})'))
    return ok

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='Training run directory')
    parser.add_argument('--kimg', type=int, required=True, help='Checkpoint kimg to verify')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for the forward/backward check')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    state_path = os.path.join(args.run_dir, f'training-state-{args.kimg:06d}.pt')
    snapshot_path = os.path.join(args.run_dir, f'network-snapshot-{args.kimg:06d}.pkl')
    stats_path = os.path.join(args.run_dir, 'stats.jsonl')

    ok = True
    ok &= check(os.path.isfile(state_path), f'training state exists: {state_path}')
    ok &= check(os.path.isfile(snapshot_path), f'network snapshot exists: {snapshot_path}')
    if not ok:
        print('VERIFY FAILED')
        return 1

    state_ok, state_net = verify_training_state(state_path)
    ok &= state_ok
    snapshot_ok, _ema, data = verify_snapshot(snapshot_path)
    ok &= snapshot_ok
    ok &= verify_forward_backward(state_net, data['dataset_kwargs'], args.batch, torch.device(args.device))
    ok &= verify_stats(stats_path)

    print('VERIFY PASSED' if ok else 'VERIFY FAILED')
    return 0 if ok else 1

#----------------------------------------------------------------------------

if __name__ == '__main__':
    sys.exit(main())

#----------------------------------------------------------------------------
