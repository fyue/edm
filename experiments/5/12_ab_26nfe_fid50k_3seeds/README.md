# Experiment 5-12: Variable-Coefficient AB 26NFE FID-50k over 3 Seed Ranges

可変係数ABを5-8の固定係数ABと同じ26 steps／26 NFEで評価し、AB係数の違いだけを比べる。FID-50k（3 seed平均）は2.45947で、固定係数（1.91000）より悪い。

## Purpose

EDM rho scheduleに合わせた可変係数Adams-Bashforth sampler `ab` を
26 steps / 26 NFEで評価する。実験5-8の固定係数PNDM no-RKとsteps、NFE、
sigma schedule、warmup、seed範囲を揃え、AB係数の違いを比較する。

| 実験 | sampler | AB係数 | steps | NFE |
|---|---|---|---:|---:|
| 5-8 | `pndm-no-rk` | 等間隔用の固定係数 | 26 | 26 |
| 5-12 | `ab` | rho scheduleに合わせた可変係数 | 26 | 26 |

両samplerともEuler、AB2、AB3で履歴を立ち上げ、その後AB4を使う。RK warmupは
使用せず、各stepでnetworkを1回評価する。主な比較要因はAB係数である。

## Conditions

| checkpoint | sampler | steps | NFE | seeds |
|---|---|---:|---:|---|
| `pretrained` | `ab` | 26 | 26 | `0-49999` |
| `pretrained` | `ab` | 26 | 26 | `50000-99999` |
| `pretrained` | `ab` | 26 | 26 | `100000-149999` |

- samples: `50000`
- sigma schedule: EDM rho schedule (`rho=7`)
- `S_churn=0`, `S_min=0`, `S_max=inf`, `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- target GPU: physical index `0`

## Plan Check

```bash
PLAN_ONLY=1 \
  experiments/5/12_ab_26nfe_fid50k_3seeds/scripts/run_ab_26nfe_fid50k_3seeds.sh
```

## Run

ローカル設定（`~/.config/edm/gpu.env` の `GPU_INDEX`・`EXPECTED_GPU_NAME`）のGPUが空くまで待機してから開始する。

```bash
tmux new -s edm-exp5-12-ab-26nfe
experiments/5/12_ab_26nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Smoke Run

```bash
tmux new -s edm-exp5-12-ab-26nfe-smoke
NUM_IMAGES=1024 \
CONFIG=experiments/5/12_ab_26nfe_fid50k_3seeds/config/ab_26nfe_smoke.tsv \
  experiments/5/12_ab_26nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor and Partial Run

```bash
tail -f experiments/5/12_ab_26nfe_fid50k_3seeds/logs/ab-26nfe-fid50k-3seeds.log
tail -f experiments/5/12_ab_26nfe_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

特定のseed範囲だけを実行する場合:

```bash
SEED_FILTER="0-49999" \
  experiments/5/12_ab_26nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

同じコマンドを再実行すると、summaryに記録済みの条件はスキップする。現在の条件が
終わったところで停止する場合は次のstop fileを作成する。

```bash
touch experiments/5/12_ab_26nfe_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Outputs

```text
experiments/5/12_ab_26nfe_fid50k_3seeds/results/ab_26nfe_fid50k_3seeds_summary.tsv
experiments/5/12_ab_26nfe_fid50k_3seeds/results/ab_26nfe_fid50k_3seeds_stats.tsv
```
