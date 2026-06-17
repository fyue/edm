# Experiment 4 Stage 2 Second: Extra Seeds for Representative FID-50k Pairs

## Purpose

Stage 2 experiment 1 の代表ペアについて、追加の2 seed範囲を評価し、3 seed範囲で `mean ± std` を出す。

代表値は最小値ではなく平均値を使う。改善幅は同じseed範囲内で

```text
improvement = FID(S_churn=0) - FID(best S_churn)
```

として計算し、その paired improvement の `mean ± std` を報告する。

## Conditions

既存の `0-49999` は experiment 1 の結果を使う。このexperimentでは追加分だけを実行する。

| checkpoint | S_churn | additional seeds |
|---|---|---|
| `5.5k` | `0, 5` | `50000-99999`, `100000-149999` |
| `10k` | `0, 3` | `50000-99999`, `100000-149999` |
| `20k` | `0, 3` | `50000-99999`, `100000-149999` |
| `40k` | `0, 2` | `50000-99999`, `100000-149999` |
| `80k` | `0, 3` | `50000-99999`, `100000-149999` |
| `120k` | `0, 3` | `50000-99999`, `100000-149999` |
| `pretrained` | `0, 3` | `50000-99999`, `100000-149999` |

合計28条件。

## Settings

- samples: `50000`
- steps: `18`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## Plan Check

```bash
PLAN_ONLY=1 experiments/4/stage2/2_fid50k_extra_seeds/scripts/run_fid50k_extra_seeds.sh
```

## Run

```bash
tmux new -s edm-exp4-stage2-extra-seeds
experiments/4/stage2/2_fid50k_extra_seeds/scripts/run_fid50k_extra_seeds.sh
```

## Monitor

```bash
tail -f experiments/4/stage2/2_fid50k_extra_seeds/logs/fid50k-extra-seeds.log
```

## Partial Runs

片方のseed範囲だけ先に実行する。

```bash
SEED_FILTER="50000-99999" \
  experiments/4/stage2/2_fid50k_extra_seeds/scripts/run_fid50k_extra_seeds.sh
```

特定checkpointだけ実行する。

```bash
CHECKPOINT_FILTER="80k 120k pretrained" \
  experiments/4/stage2/2_fid50k_extra_seeds/scripts/run_fid50k_extra_seeds.sh
```

## Resume / Stop

同じコマンドを再実行すると、`fid50k_extra_seeds_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成をスキップして、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/4/stage2/2_fid50k_extra_seeds/STOP_AFTER_CURRENT_CONDITION
```

## Summarize 3 Seed Ranges

追加seed実験が終わったら、experiment 1 の `0-49999` とこのexperimentの追加2 seedを結合して集計する。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/4/stage2/2_fid50k_extra_seeds/scripts/summarize_fid50k_3seed.py
```

Outputs:

```text
experiments/4/stage2/2_fid50k_extra_seeds/results/fid50k_3seed_condition_stats.tsv
experiments/4/stage2/2_fid50k_extra_seeds/results/fid50k_3seed_improvement_stats.tsv
experiments/4/stage2/2_fid50k_extra_seeds/results/fid50k_3seed_improvement.png
```

## Runtime Estimate

1条件あたり約70-75分。28条件なので、直列では約35時間前後を見込む。
