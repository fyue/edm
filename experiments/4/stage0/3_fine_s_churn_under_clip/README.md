# Experiment 4 Stage 0 Third: Fine S_churn Sweep Under Clip

## 目的

Stage 0 の3番目の実験として、`S_churn=20` と `S_churn=40` が同じ結果になった理由を踏まえ、クリップ閾値未満の `S_churn` を細かく調べる。

EDM samplerでは次の式で一時的なノイズ追加量が決まる。

```python
gamma = min(S_churn / num_steps, sqrt(2) - 1)
```

今回の `steps=18` では、`S_churn >= 7.46` で `gamma` が飽和する。そのため、この実験では `S_churn=0,1,2,4,5,6,7` を見る。

## 対象

デフォルトでは次の4モデルを評価する。

| checkpoint | network |
|---|---|
| `10k` | `network-snapshot-010000.pkl` |
| `40k` | `network-snapshot-040000.pkl` |
| `120k` | `network-snapshot-120000.pkl` |
| `pretrained` | official CIFAR-10 conditional EDM |

checkpoint一覧は `experiments/4/stage0/3_fine_s_churn_under_clip/config/stage0_checkpoints.tsv` に置いている。

## 実行条件

- `S_churn`: `0, 1, 2, 4, 5, 6, 7`
- samples: `1024`
- seeds: `0-1023`
- steps: `18`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## 実行方法

実行前に予定だけ確認する。

```bash
PLAN_ONLY=1 experiments/4/stage0/3_fine_s_churn_under_clip/scripts/run_stage0_fine_churn.sh
```

tmuxで実行する。

```bash
tmux new -s edm-exp4-stage0-fine-churn
experiments/4/stage0/3_fine_s_churn_under_clip/scripts/run_stage0_fine_churn.sh
```

進捗確認。

```bash
tail -f experiments/4/stage0/3_fine_s_churn_under_clip/logs/stage0-fine-churn.log
```

結果確認。

```bash
cat experiments/4/stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_summary.tsv
```

FID-`S_churn` 曲線を描く。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/4/stage0/3_fine_s_churn_under_clip/scripts/plot_fine_churn_curve.py
```

出力:

```text
experiments/4/stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_curve.png
```

## 一部だけ実行

まず `10k, 120k, pretrained` だけを見たい場合:

```bash
CHECKPOINT_FILTER="10k 120k pretrained" \
  experiments/4/stage0/3_fine_s_churn_under_clip/scripts/run_stage0_fine_churn.sh
```

`S_churn=2,4,6` だけを追加確認したい場合:

```bash
CHURNS="2 4 6" \
  experiments/4/stage0/3_fine_s_churn_under_clip/scripts/run_stage0_fine_churn.sh
```

## 再開

同じコマンドを再実行すると、`stage0_fine_churn_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成もスキップして、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/4/stage0/3_fine_s_churn_under_clip/STOP_AFTER_CURRENT_CONDITION
```

## 出力

```text
experiments/4/stage0/3_fine_s_churn_under_clip/images/<checkpoint>/S_churn-*/n1024-seeds-0-1023/
experiments/4/stage0/3_fine_s_churn_under_clip/logs/stage0-*.log
experiments/4/stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_summary.tsv
experiments/4/stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_curve.png
```

## 解釈上の注意

- `FID-1k` はサンプル数が少ないため、絶対値としては不安定。
- 同じ `NUM_IMAGES`, `SEEDS`, `steps`, `S_min`, `S_max`, `S_noise` で揃えた条件間の相対比較として使う。
- この実験では飽和前の形を見ることが目的なので、`S_churn=20,40` はデフォルトから外している。
