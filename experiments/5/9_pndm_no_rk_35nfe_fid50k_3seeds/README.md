# Experiment 5-9: PNDM no-RK 35NFE FID-50k over 3 Seed Ranges

## Purpose

実験5-8の `pndm-no-rk` を `steps=35`, `NFE=35` に増やし、FID-50kを3つのseed範囲で評価する。

実験5-8は `steps=26`, `NFE=26` だった。今回は同じ no-RK warmup のままstep数を35に増やし、EDM標準やAB 35NFEと同じNFE予算で比較する。

## Comparison Targets

| 実験 | sampler | steps | NFE | 備考 |
|---|---|---:|---:|---|
| 5-5 | `pndm` | 26 | 35 | RK warmupあり |
| 5-7 | `ab` | 35 | 35 | 可変係数Adams-Bashforth |
| 5-8 | `pndm-no-rk` | 26 | 26 | RK warmupなし |
| 5-9 | `pndm-no-rk` | 35 | 35 | 今回の条件 |

## Conditions

| checkpoint | sampler | steps | NFE | seeds |
|---|---|---:|---:|---|
| `pretrained` | `pndm-no-rk` | 35 | 35 | `0-49999` |
| `pretrained` | `pndm-no-rk` | 35 | 35 | `50000-99999` |
| `pretrained` | `pndm-no-rk` | 35 | 35 | `100000-149999` |

- samples: `50000`
- `S_churn=0`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- GPU: physical index `0`, `Quadro RTX 8000`

## Plan Check

```bash
PLAN_ONLY=1 \
  experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/scripts/run_pndm_no_rk_35nfe_fid50k_3seeds.sh
```

## Run

GPU index 0が空いた後に開始する。

```bash
tmux new -s edm-exp5-9-pndm-no-rk-35nfe-fid50k-scheduled
experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Smoke Run

```bash
tmux new -s edm-exp5-9-pndm-no-rk-35nfe-smoke-scheduled
NUM_IMAGES=1024 \
CONFIG=experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/config/pndm_no_rk_35nfe_smoke.tsv \
  experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor

```bash
tail -f experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/logs/pndm-no-rk-35nfe-fid50k-3seeds.log
tail -f experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

## Partial Run

```bash
SEED_FILTER="0-49999" \
  experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Resume / Stop

summaryに記録済みの条件は再実行時にスキップする。画像が50000枚そろっている場合は生成をスキップし、FID計算から再開する。

```bash
touch experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Outputs

```text
experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/results/pndm_no_rk_35nfe_fid50k_3seeds_summary.tsv
experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds/results/pndm_no_rk_35nfe_fid50k_3seeds_stats.tsv
```
