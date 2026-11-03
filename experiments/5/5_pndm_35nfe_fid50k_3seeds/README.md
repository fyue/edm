# Experiment 5-5: PNDM Sampler 35NFE FID-50k over 3 Seed Ranges

5-1のPNDMをsteps=26にしてEDM Heunと同じ35 NFEで評価する。FID-50k（3 seed平均）は1.91047で、EDM Heunの1.83906に届かない。

## Purpose

`generate.py --sampler=pndm` の35NFE条件での FID-50k を、3つの seed 範囲で評価する。

PNDM実装は最初の3区間でRK4ウォームアップを使い、それ以降は1区間1評価で進む。
そのため `steps=N` の network evaluation 回数は `N + 9` となる。35NFEに合わせるため、
この実験では `steps=26` を使う。

FID は seed によるばらつきがあるため、各条件について次を出す。

- seed別 FID
- `mean`
- sample `std`
- `min`

## Default Condition

まずは CIFAR-10 conditional pretrained model の PNDM sampler を評価する。

| checkpoint | sampler | S_churn | seeds |
|---|---|---:|---|
| `pretrained` | `pndm` | `0` | `0-49999` |
| `pretrained` | `pndm` | `0` | `50000-99999` |
| `pretrained` | `pndm` | `0` | `100000-149999` |

追加の checkpoint を評価したい場合は、`config/pndm_35nfe_fid50k_3seeds.tsv` に同じ3 seed範囲の行を追加する。

## Settings

- samples: `50000`
- steps: `26`
- NFE: `35`
- sampler: `pndm`
- `S_churn=0`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## Plan Check

```bash
PLAN_ONLY=1 experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/run_pndm_35nfe_fid50k_3seeds.sh
```

## Run

```bash
tmux new -s edm-exp5-5-pndm-35nfe-fid50k
experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/run_pndm_35nfe_fid50k_3seeds.sh
```

対象GPUが既に使われている場合は、待機スケジューラをtmuxで起動する。ローカル設定（`~/.config/edm/gpu.env` の `GPU_INDEX`・`EXPECTED_GPU_NAME`）のGPU上の compute process がなくなったら自動で本実験を開始する。

```bash
tmux new -s edm-exp5-5-pndm-35nfe-fid50k-scheduled
experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

小さな動作確認を、起動中の実験終了後に開始する場合:

```bash
tmux new -s edm-exp5-5-pndm-35nfe-smoke-scheduled
NUM_IMAGES=1024 \
CONFIG=experiments/5/5_pndm_35nfe_fid50k_3seeds/config/pndm_35nfe_smoke.tsv \
  experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Monitor

```bash
tail -f experiments/5/5_pndm_35nfe_fid50k_3seeds/logs/pndm-35nfe-fid50k-3seeds.log
```

スケジューラの待機ログを見る場合:

```bash
tail -f experiments/5/5_pndm_35nfe_fid50k_3seeds/logs/schedule-after-gpu-free.log
```

対象GPUはローカル設定（`~/.config/edm/gpu.env` の `GPU_INDEX`・`EXPECTED_GPU_NAME`）で決まり、コマンドラインの環境変数では変更しない。別の設定ファイルは `EDM_GPU_CONFIG` で指定する。確認間隔は `CHECK_INTERVAL_SEC` で変更できる。

```bash
CHECK_INTERVAL_SEC=60 \
  experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## Partial Runs

特定 seed 範囲だけ実行する。

```bash
SEED_FILTER="0-49999" \
  experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/run_pndm_35nfe_fid50k_3seeds.sh
```

特定 checkpoint だけ実行する。

```bash
CHECKPOINT_FILTER="pretrained" \
  experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/run_pndm_35nfe_fid50k_3seeds.sh
```

小さく動作確認する。

```bash
NUM_IMAGES=1024 \
CONFIG=experiments/5/5_pndm_35nfe_fid50k_3seeds/config/pndm_35nfe_smoke.tsv \
  experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/run_pndm_35nfe_fid50k_3seeds.sh
```

## Resume / Stop

同じコマンドを再実行すると、`pndm_35nfe_fid50k_3seeds_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成をスキップし、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/5/5_pndm_35nfe_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Summarize

実行スクリプトは最後に自動で集計を試みる。手動で集計し直す場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/5/5_pndm_35nfe_fid50k_3seeds/scripts/summarize_pndm_35nfe_fid50k_3seeds.py
```

Outputs:

```text
experiments/5/5_pndm_35nfe_fid50k_3seeds/results/pndm_35nfe_fid50k_3seeds_summary.tsv
experiments/5/5_pndm_35nfe_fid50k_3seeds/results/pndm_35nfe_fid50k_3seeds_stats.tsv
```

`pndm_35nfe_fid50k_3seeds_stats.tsv` には、1024枚の動作確認（checkpoint `pretrained-smoke`、FID 30.1815）の行も含まれる。本評価の値は `pretrained` の行（FID-50k×3 seed）を使う。
