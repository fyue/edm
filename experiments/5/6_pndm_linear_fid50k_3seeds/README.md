# Experiment 5-6: PNDM Fixed AB4 with Linear Sigma Schedule FID-50k over 3 Seed Ranges

## Purpose

`generate.py --sampler=pndm-linear` の FID-50k を、3つの seed 範囲で評価する。

この sampler は PNDM/F-PNDM風の固定係数AB4更新は変えず、sigma schedule だけを `sigma_max` から `0` まで等間隔にする。通常の `pndm` は EDM rho schedule を使うため、固定AB4の等間隔step仮定とのズレがある。この実験では、そのズレをschedule側で取り除いた場合のFIDを見る。

FID は seed によるばらつきがあるため、各条件について次を出す。

- seed別 FID
- `mean`
- sample `std`
- `min`

## Default Condition

まずは CIFAR-10 conditional pretrained model の `pndm-linear` sampler を評価する。

| checkpoint | sampler | S_churn | seeds |
|---|---|---:|---|
| `pretrained` | `pndm-linear` | `0` | `0-49999` |
| `pretrained` | `pndm-linear` | `0` | `50000-99999` |
| `pretrained` | `pndm-linear` | `0` | `100000-149999` |

追加の checkpoint を評価したい場合は、`config/pndm_linear_fid50k_3seeds.tsv` に同じ3 seed範囲の行を追加する。

## Settings

- samples: `50000`
- steps: `18`
- sampler: `pndm-linear`
- schedule: linear sigma from `sigma_max` to `0`
- `S_churn=0`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## Plan Check

```bash
PLAN_ONLY=1 experiments/5/6_pndm_linear_fid50k_3seeds/scripts/run_pndm_linear_fid50k_3seeds.sh
```

## Run

```bash
tmux new -s edm-exp5-6-pndm-linear-fid50k
experiments/5/6_pndm_linear_fid50k_3seeds/scripts/run_pndm_linear_fid50k_3seeds.sh
```

index 0 の Quadro RTX 8000 が既に使われている場合は、待機スケジューラをtmuxで起動する。
GPU index 0上の compute process がなくなったら自動で本実験を開始する。

```bash
tmux new -s edm-exp5-6-pndm-linear-fid50k-scheduled
experiments/5/6_pndm_linear_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

小さな動作確認を、起動中の実験終了後に開始する場合:

```bash
tmux new -s edm-exp5-6-pndm-linear-smoke-scheduled
NUM_IMAGES=1024 \
CONFIG=experiments/5/6_pndm_linear_fid50k_3seeds/config/pndm_linear_smoke.tsv \
  experiments/5/6_pndm_linear_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor

```bash
tail -f experiments/5/6_pndm_linear_fid50k_3seeds/logs/pndm-linear-fid50k-3seeds.log
```

スケジューラの待機ログを見る場合:

```bash
tail -f experiments/5/6_pndm_linear_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

待機間隔や対象GPUは環境変数で変更できる。既定では `GPU_INDEX=0` を見て、
そのGPU名が `Quadro RTX 8000` であることを確認する。

```bash
GPU_INDEX=0 EXPECTED_GPU_NAME="Quadro RTX 8000" CHECK_INTERVAL_SEC=60 \
  experiments/5/6_pndm_linear_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Partial Runs

特定 seed 範囲だけ実行する。

```bash
SEED_FILTER="0-49999" \
  experiments/5/6_pndm_linear_fid50k_3seeds/scripts/run_pndm_linear_fid50k_3seeds.sh
```

特定 checkpoint だけ実行する。

```bash
CHECKPOINT_FILTER="pretrained" \
  experiments/5/6_pndm_linear_fid50k_3seeds/scripts/run_pndm_linear_fid50k_3seeds.sh
```

小さく動作確認する。

```bash
NUM_IMAGES=1024 \
CONFIG=experiments/5/6_pndm_linear_fid50k_3seeds/config/pndm_linear_smoke.tsv \
  experiments/5/6_pndm_linear_fid50k_3seeds/scripts/run_pndm_linear_fid50k_3seeds.sh
```

## Resume / Stop

同じコマンドを再実行すると、`pndm_linear_fid50k_3seeds_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成をスキップし、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/5/6_pndm_linear_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Summarize

実行スクリプトは最後に自動で集計を試みる。手動で集計し直す場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/5/6_pndm_linear_fid50k_3seeds/scripts/summarize_pndm_linear_fid50k_3seeds.py
```

Outputs:

```text
experiments/5/6_pndm_linear_fid50k_3seeds/results/pndm_linear_fid50k_3seeds_summary.tsv
experiments/5/6_pndm_linear_fid50k_3seeds/results/pndm_linear_fid50k_3seeds_stats.tsv
```
