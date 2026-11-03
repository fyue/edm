# Experiment 5-7: No-RK Warmup AB 35NFE FID-50k over 3 Seed Ranges

5-4の可変係数ABを35 steps／35 NFEに増やして評価する。FID-50k（3 seed平均）は2.12443で、18 NFEの3.57748から改善したが、同じ35 NFEのPNDM（1.91047）やEDM Heun（1.83906）より悪い。

## Purpose

`generate.py --sampler=ab`を35 steps / 35NFEで実行し、FID-50kを3つの
seed範囲で評価する。

このsamplerは最初のRK warmupを使わず、微分履歴の本数に応じて
Euler, AB2, AB3, AB4と次数を上げる。各stepのnetwork評価は1回なので、
`steps=NFE`となる。EDMのrho scheduleは非等間隔のため、Adams-Bashforth
係数は各区間でLagrange補間から計算する。

実験5-4の18 steps / 18NFEからstep数を増やし、同じ35NFE予算で次と比較する。

- 実験5-5: RK4 warmup PNDM、26 steps / 35NFE
- EDM標準: Heun、18 steps / 35NFE

## Default Condition

| checkpoint | sampler | S_churn | seeds |
|---|---|---:|---|
| `pretrained` | `ab` | `0` | `0-49999` |
| `pretrained` | `ab` | `0` | `50000-99999` |
| `pretrained` | `ab` | `0` | `100000-149999` |

## Settings

- samples: `50000`
- steps: `35`
- NFE: `35`
- sampler: `ab`
- warmup: Euler, AB2, AB3; no RK
- main solver: variable-step AB4
- sigma schedule: EDM rho schedule (`rho=7`)
- `S_churn=0`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- target GPU: physical index `0`, `Quadro RTX 8000`

## Plan Check

```bash
PLAN_ONLY=1 \
  experiments/5/7_ab_35nfe_fid50k_3seeds/scripts/run_ab_35nfe_fid50k_3seeds.sh
```

## Run

GPU index 0が空くまで待機し、その後に実験を開始する。

```bash
tmux new -s edm-exp5-7-ab-35nfe-fid50k-scheduled
experiments/5/7_ab_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

待機せず直接開始する場合:

```bash
tmux new -s edm-exp5-7-ab-35nfe-fid50k
experiments/5/7_ab_35nfe_fid50k_3seeds/scripts/run_ab_35nfe_fid50k_3seeds.sh
```

## Smoke Run

GPUが空いた後に1024枚で確認する。

```bash
tmux new -s edm-exp5-7-ab-35nfe-smoke-scheduled
NUM_IMAGES=1024 \
CONFIG=experiments/5/7_ab_35nfe_fid50k_3seeds/config/ab_35nfe_smoke.tsv \
  experiments/5/7_ab_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor

```bash
tail -f experiments/5/7_ab_35nfe_fid50k_3seeds/logs/ab-35nfe-fid50k-3seeds.log
tail -f experiments/5/7_ab_35nfe_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

## Partial Runs

特定seed範囲だけ実行する。

```bash
SEED_FILTER="0-49999" \
  experiments/5/7_ab_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Resume / Stop

同じコマンドを再実行すると、summaryに記録済みの条件はスキップする。
画像が既に十分ある場合は生成をスキップし、FID計算から再開する。

現在の条件が終わったところで止める場合:

```bash
touch experiments/5/7_ab_35nfe_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Summarize

実行終了時に自動集計する。手動で集計し直す場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/5/7_ab_35nfe_fid50k_3seeds/scripts/summarize_ab_35nfe_fid50k_3seeds.py
```

Outputs:

```text
experiments/5/7_ab_35nfe_fid50k_3seeds/results/ab_35nfe_fid50k_3seeds_summary.tsv
experiments/5/7_ab_35nfe_fid50k_3seeds/results/ab_35nfe_fid50k_3seeds_stats.tsv
```
