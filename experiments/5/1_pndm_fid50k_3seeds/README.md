# Experiment 5-1: PNDM Sampler FID-50k over 3 Seed Ranges

## Purpose

`generate.py --sampler=pndm` の FID-50k を、3つの seed 範囲で評価する。

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

追加の checkpoint を評価したい場合は、`config/pndm_fid50k_3seeds.tsv` に同じ3 seed範囲の行を追加する。

## Settings

- samples: `50000`
- steps: `18`
- sampler: `pndm`
- `S_churn=0`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## Plan Check

```bash
PLAN_ONLY=1 experiments/5/1_pndm_fid50k_3seeds/scripts/run_pndm_fid50k_3seeds.sh
```

## Run

```bash
tmux new -s edm-exp5-1-pndm-fid50k
experiments/5/1_pndm_fid50k_3seeds/scripts/run_pndm_fid50k_3seeds.sh
```

## Monitor

```bash
tail -f experiments/5/1_pndm_fid50k_3seeds/logs/pndm-fid50k-3seeds.log
```

## Partial Runs

特定 seed 範囲だけ実行する。

```bash
SEED_FILTER="0-49999" \
  experiments/5/1_pndm_fid50k_3seeds/scripts/run_pndm_fid50k_3seeds.sh
```

特定 checkpoint だけ実行する。

```bash
CHECKPOINT_FILTER="pretrained" \
  experiments/5/1_pndm_fid50k_3seeds/scripts/run_pndm_fid50k_3seeds.sh
```

小さく動作確認する。

```bash
NUM_IMAGES=1024 \
CONFIG=experiments/5/1_pndm_fid50k_3seeds/config/pndm_smoke.tsv \
  experiments/5/1_pndm_fid50k_3seeds/scripts/run_pndm_fid50k_3seeds.sh
```

## Resume / Stop

同じコマンドを再実行すると、`pndm_fid50k_3seeds_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成をスキップし、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/5/1_pndm_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

## Summarize

実行スクリプトは最後に自動で集計を試みる。手動で集計し直す場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/5/1_pndm_fid50k_3seeds/scripts/summarize_pndm_fid50k_3seeds.py
```

Outputs:

```text
experiments/5/1_pndm_fid50k_3seeds/results/pndm_fid50k_3seeds_summary.tsv
experiments/5/1_pndm_fid50k_3seeds/results/pndm_fid50k_3seeds_stats.tsv
```
