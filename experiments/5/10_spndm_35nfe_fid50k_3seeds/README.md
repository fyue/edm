# Experiment 5-10: S-PNDM Sampler 35NFE FID-50k over 3 Seed Ranges

5-2のS-PNDMをsteps=34にして35 NFEで評価する。FID-50k（3 seed平均）は2.21971で、19 NFEの3.14766から改善した。

## Purpose

実験5-2を基準に、`generate.py --sampler=spndm` の35NFE条件でFID-50kを
3つのseed範囲について評価する。

S-PNDM実装は各stepの始点でnetworkを1回評価し、最初のstepだけHeun補正の
ために終点でも1回評価する。そのため `steps=N` のnetwork evaluation回数は
`N + 1` となる。35NFEに合わせるため、この実験では `steps=34` を使う。

各条件についてseed別FIDと、その `mean`、sample `std`、`min` を出力する。

## Default Condition

| checkpoint | sampler | S_churn | seeds |
|---|---|---:|---|
| `pretrained` | `spndm` | `0` | `0-49999` |
| `pretrained` | `spndm` | `0` | `50000-99999` |
| `pretrained` | `spndm` | `0` | `100000-149999` |

追加のcheckpointを評価する場合は、
`config/spndm_35nfe_fid50k_3seeds.tsv` に同じ3 seed範囲の行を追加する。

## Settings

- samples: `50000`
- steps: `34`
- NFE: `35`
- sampler: `spndm`
- `S_churn=0`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## Plan Check

```bash
PLAN_ONLY=1 \
  experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/run_spndm_35nfe_fid50k_3seeds.sh
```

## Run

GPU index 0が空くまで待機してから開始する。

```bash
tmux new -s edm-exp5-10-spndm-35nfe-fid50k
experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

既にジョブスケジューラからGPUを排他的に割り当てられている場合は、実行本体を
直接起動できる。

```bash
experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/run_spndm_35nfe_fid50k_3seeds.sh
```

## Smoke Test

GPUが空いてから1024枚で動作確認する。

```bash
NUM_IMAGES=1024 \
CONFIG=experiments/5/10_spndm_35nfe_fid50k_3seeds/config/spndm_35nfe_smoke.tsv \
  experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor

```bash
tail -f experiments/5/10_spndm_35nfe_fid50k_3seeds/logs/spndm-35nfe-fid50k-3seeds.log
tail -f experiments/5/10_spndm_35nfe_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

待機対象のGPUとGPU名の検証はローカル設定（`~/.config/edm/gpu.env` の `GPU_INDEX`・`EXPECTED_GPU_NAME`）で決まる。確認間隔は
`CHECK_INTERVAL_SEC` で変更できる。

## Partial Runs

```bash
SEED_FILTER="0-49999" \
  experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

```bash
CHECKPOINT_FILTER="pretrained" \
  experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Resume / Stop

同じコマンドを再実行すると、summaryに記録済みの条件はスキップする。画像が
既に必要枚数ある場合は生成をスキップし、FID計算から再開する。

現在の条件が終了した時点で停止するには、別ターミナルで次を実行する。

```bash
touch experiments/5/10_spndm_35nfe_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Summarize

実行終了時に自動集計する。手動で再集計する場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/5/10_spndm_35nfe_fid50k_3seeds/scripts/summarize_spndm_35nfe_fid50k_3seeds.py
```

Outputs:

```text
experiments/5/10_spndm_35nfe_fid50k_3seeds/results/spndm_35nfe_fid50k_3seeds_summary.tsv
experiments/5/10_spndm_35nfe_fid50k_3seeds/results/spndm_35nfe_fid50k_3seeds_stats.tsv
```
