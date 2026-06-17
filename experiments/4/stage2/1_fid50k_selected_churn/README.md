# Experiment 4 Stage 2 First: Selected S_churn FID-50k

## Purpose

Stage 1 の `FID-5k` で見えた傾向を、正式評価に近い `FID-50k` で確認する。

主に確認したい仮説は次の2点。

- 未学習寄りの checkpoint ほど `S_churn` による改善が大きい。
- 十分訓練された checkpoint や pretrained では `S_churn` の改善が小さい、またはノイズ程度になる。

## Conditions

最初から全探索はせず、Stage 1で意味がありそうだった代表条件だけを回す。

| checkpoint | S_churn |
|---|---|
| `5.5k` | `0, 3, 5` |
| `10k` | `0, 2, 3` |
| `20k` | `0, 3, 5` |
| `40k` | `0, 2, 3, 5` |
| `80k` | `0, 3` |
| `120k` | `0, 3` |
| `pretrained` | `0, 3, 5` |

合計20条件。

## Runtime Estimate

Stage 1 のログでは `5000` samples の生成が約7分だったため、`50000` samples は1条件あたりおよそ70-75分を見込む。

20条件を直列で回す場合の目安は約24-26時間。

## Settings

- samples: `50000`
- seeds: `0-49999`
- steps: `18`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

`50000-99999` や `100000-149999` はこのStage2初手には入れない。必要になったら、代表条件だけ後から追加する。

## Plan Check

```bash
PLAN_ONLY=1 experiments/4/stage2/1_fid50k_selected_churn/scripts/run_fid50k_selected_churn.sh
```

## Run

```bash
tmux new -s edm-exp4-stage2-fid50k
experiments/4/stage2/1_fid50k_selected_churn/scripts/run_fid50k_selected_churn.sh
```

## Monitor

```bash
tail -f experiments/4/stage2/1_fid50k_selected_churn/logs/fid50k-selected-churn.log
```

## Plot

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/4/stage2/1_fid50k_selected_churn/scripts/plot_fid50k_selected_churn.py
```

Outputs:

```text
experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_curve.png
experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_improvement.png
```

## Partial Runs

特定checkpointだけ実行する。

```bash
CHECKPOINT_FILTER="5.5k 10k" \
  experiments/4/stage2/1_fid50k_selected_churn/scripts/run_fid50k_selected_churn.sh
```

まず小さく動作確認する。

```bash
NUM_IMAGES=1024 SEEDS=0-1023 \
  experiments/4/stage2/1_fid50k_selected_churn/scripts/run_fid50k_selected_churn.sh
```

## Resume / Stop

同じコマンドを再実行すると、`fid50k_selected_churn_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成もスキップして、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/4/stage2/1_fid50k_selected_churn/STOP_AFTER_CURRENT_CONDITION
```

## Outputs

```text
experiments/4/stage2/1_fid50k_selected_churn/images/<checkpoint>/S_churn-*/n50000-seeds-0-49999/
experiments/4/stage2/1_fid50k_selected_churn/logs/fid50k-*.log
experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_summary.tsv
experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_curve.png
experiments/4/stage2/1_fid50k_selected_churn/results/fid50k_selected_churn_improvement.png
```
