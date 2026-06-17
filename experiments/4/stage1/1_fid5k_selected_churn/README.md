# Experiment 4 Stage 1 First: Selected S_churn FID-5k

## 目的

Stage 0 の FID-1k で見えた傾向を、少し信頼度の高い `FID-5k` で確認する。

対象条件は次の通り。

| checkpoint | S_churn |
|---|---|
| `5.5k` | `0, 1, 2, 3, 4, 5` |
| `10k` | `0, 1, 2, 3, 4, 5` |
| `15.0k` | `0, 1, 2, 3, 4, 5` |
| `20k` | `0, 1, 2, 3, 4, 5` |
| `40k` | `0, 1, 2, 3, 4, 5` |
| `80k` | `0, 1, 2, 3, 4, 5` |
| `120k` | `0, 1, 2, 3, 4, 5` |
| `pretrained` | `0, 1, 2, 3, 4, 5` |

既存の15条件は `summary.tsv` に記録済みなので、同じスクリプトを再実行すると追加分だけ走る。`5.5k` と `15.0k` は、厳密にはそれぞれ `network-snapshot-005509.pkl` と `network-snapshot-015018.pkl` を使う。

## 実行条件

- samples: `5000`
- seeds: `0-4999`
- steps: `18`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## 実行方法

実行前に予定だけ確認する。

```bash
PLAN_ONLY=1 experiments/4/stage1/1_fid5k_selected_churn/scripts/run_fid5k_selected_churn.sh
```

tmuxで実行する。

```bash
tmux new -s edm-exp4-stage1-fid5k
experiments/4/stage1/1_fid5k_selected_churn/scripts/run_fid5k_selected_churn.sh
```

進捗確認。

```bash
tail -f experiments/4/stage1/1_fid5k_selected_churn/logs/fid5k-selected-churn.log
```

結果確認。

```bash
cat experiments/4/stage1/1_fid5k_selected_churn/results/fid5k_selected_churn_summary.tsv
```

FID-`S_churn` 曲線を描く。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/4/stage1/1_fid5k_selected_churn/scripts/plot_fid5k_selected_churn.py
```

出力:

```text
experiments/4/stage1/1_fid5k_selected_churn/results/fid5k_selected_churn_curve.png
```

## 一部だけ実行

特定checkpointだけ実行する。

```bash
CHECKPOINT_FILTER="10k 120k" \
  experiments/4/stage1/1_fid5k_selected_churn/scripts/run_fid5k_selected_churn.sh
```

サンプル数を変えて試す。

```bash
NUM_IMAGES=2048 SEEDS=0-2047 \
  experiments/4/stage1/1_fid5k_selected_churn/scripts/run_fid5k_selected_churn.sh
```

## 再開

同じコマンドを再実行すると、`fid5k_selected_churn_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成もスキップして、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/4/stage1/1_fid5k_selected_churn/STOP_AFTER_CURRENT_CONDITION
```

## 出力

```text
experiments/4/stage1/1_fid5k_selected_churn/images/<checkpoint>/S_churn-*/n5000-seeds-0-4999/
experiments/4/stage1/1_fid5k_selected_churn/logs/fid5k-*.log
experiments/4/stage1/1_fid5k_selected_churn/results/fid5k_selected_churn_summary.tsv
experiments/4/stage1/1_fid5k_selected_churn/results/fid5k_selected_churn_curve.png
```

## 解釈上の注意

- `FID-5k` は `FID-1k` より安定するが、正式な50k FIDではない。
- 同じ `NUM_IMAGES`, `SEEDS`, `steps`, `S_min`, `S_max`, `S_noise` で揃えた条件間の相対比較として使う。
- この結果で残った候補だけを、必要なら50k FIDに進める。
