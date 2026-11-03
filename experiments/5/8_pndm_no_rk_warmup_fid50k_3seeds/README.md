# Experiment 5-8: PNDM without RK Warmup

5-5のPNDMからRK4 warmupだけを除く（固定係数AB、26 steps／26 NFE）。FID-50k（3 seed平均）は1.91000で、35 NFEの5-5（1.91047）と同等だった。

## Purpose

実験5-5のPNDMから最初の3区間のRK4 warmupだけを除き、FID-50kを3つの
seed範囲で評価する。no-RK版はEuler, AB2, AB3で履歴を作り、その後は
実験5-5と同じ固定係数AB4を使う。

## Comparison with Experiment 5-5

同じ条件:

- checkpointとseed範囲
- `steps=26`とEDM rho schedule (`rho=7`)
- 固定係数AB2, AB3, AB4
- `S_churn=0`, `S_min=0`, `S_max=inf`, `S_noise=1`

変更点はRK4 warmupの有無だけ。実験5-5は最初の3区間に各4NFEを使うため
35NFEだが、5-8は各step 1NFEなので26NFEになる。NFEを35へ戻すためにstepを
増やすとsigma gridも変わるため、この比較では増やさない。

## Conditions

| checkpoint | sampler | steps | NFE | seeds |
|---|---|---:|---:|---|
| `pretrained` | `pndm-no-rk` | 26 | 26 | `0-49999` |
| `pretrained` | `pndm-no-rk` | 26 | 26 | `50000-99999` |
| `pretrained` | `pndm-no-rk` | 26 | 26 | `100000-149999` |

- samples: `50000`
- FID reference: `fid-refs/cifar10-32x32.npz`
- GPU: physical index `0`, `Quadro RTX 8000`

## Plan Check

```bash
PLAN_ONLY=1 \
  experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/scripts/run_pndm_no_rk_fid50k_3seeds.sh
```

## Run

GPU index 0が空いた後に開始する。

```bash
tmux new -s edm-exp5-8-pndm-no-rk-fid50k-scheduled
experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Smoke Run

```bash
tmux new -s edm-exp5-8-pndm-no-rk-smoke-scheduled
NUM_IMAGES=1024 \
CONFIG=experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/config/pndm_no_rk_smoke.tsv \
  experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor

```bash
tail -f experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/logs/pndm-no-rk-fid50k-3seeds.log
tail -f experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

## Partial Run

```bash
SEED_FILTER="0-49999" \
  experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Resume / Stop

summaryに記録済みの条件は再実行時にスキップする。画像が50000枚そろっている
場合は生成をスキップし、FID計算から再開する。

```bash
touch experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Outputs

```text
experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/results/pndm_no_rk_fid50k_3seeds_summary.tsv
experiments/5/8_pndm_no_rk_warmup_fid50k_3seeds/results/pndm_no_rk_fid50k_3seeds_stats.tsv
```
