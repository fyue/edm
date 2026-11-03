# Experiment 5-11: PNDM no-RK 18NFE FID-50k over 3 Seed Ranges

固定係数AB（PNDM no-RK）を5-4の可変係数ABと同じ18 steps／18 NFEで評価し、AB係数の違いだけを比べる。FID-50k（3 seed平均）は1.99106で、可変係数（3.57748）より大幅に良い。

## Purpose

固定係数Adams-Bashforthを使う `pndm-no-rk` を18 steps / 18 NFEで評価する。
実験5-4の可変係数ABとsteps、NFE、sigma schedule、warmup、seed範囲を揃え、
非等間隔のEDM rho scheduleに対するAB係数の違いを比較する。

| 実験 | sampler | AB係数 | steps | NFE |
|---|---|---|---:|---:|
| 5-4 | `ab` | rho scheduleに合わせた可変係数 | 18 | 18 |
| 5-11 | `pndm-no-rk` | 等間隔用の固定係数 | 18 | 18 |

両samplerともEuler、AB2、AB3で履歴を立ち上げ、その後AB4を使う。RK warmupは
使用せず、各stepでnetworkを1回評価する。主な比較要因はAB係数である。

## Conditions

| checkpoint | sampler | steps | NFE | seeds |
|---|---|---:|---:|---|
| `pretrained` | `pndm-no-rk` | 18 | 18 | `0-49999` |
| `pretrained` | `pndm-no-rk` | 18 | 18 | `50000-99999` |
| `pretrained` | `pndm-no-rk` | 18 | 18 | `100000-149999` |

- samples: `50000`
- sigma schedule: EDM rho schedule (`rho=7`)
- `S_churn=0`, `S_min=0`, `S_max=inf`, `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- target GPU: physical index `0`

## Plan Check

```bash
PLAN_ONLY=1 \
  experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/scripts/run_pndm_no_rk_18nfe_fid50k_3seeds.sh
```

## Run

ローカル設定（`~/.config/edm/gpu.env` の `GPU_INDEX`・`EXPECTED_GPU_NAME`）のGPUが空くまで待機してから開始する。

```bash
tmux new -s edm-exp5-11-pndm-no-rk-18nfe
experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Smoke Run

```bash
tmux new -s edm-exp5-11-pndm-no-rk-18nfe-smoke
NUM_IMAGES=1024 \
CONFIG=experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/config/pndm_no_rk_18nfe_smoke.tsv \
  experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor and Partial Run

```bash
tail -f experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/logs/pndm-no-rk-18nfe-fid50k-3seeds.log
tail -f experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

特定のseed範囲だけを実行する場合:

```bash
SEED_FILTER="0-49999" \
  experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

同じコマンドを再実行すると、summaryに記録済みの条件はスキップする。現在の条件が
終わったところで停止する場合は次のstop fileを作成する。

```bash
touch experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Outputs

```text
experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/results/pndm_no_rk_18nfe_fid50k_3seeds_summary.tsv
experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds/results/pndm_no_rk_18nfe_fid50k_3seeds_stats.tsv
```
