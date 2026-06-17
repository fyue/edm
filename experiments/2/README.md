# Experiment 2: 120k/200k conditional generation check

## 目的

本番学習済みの CIFAR-10 class-conditional EDM checkpoint について、`generate.py` の `--class` 指定が問題なく効くかを確認する。

対象 checkpoint:

- `training-runs/cifar10-cond-ddpmpp-edm-managed/network-snapshot-120000.pkl`
- `training-runs/cifar10-cond-ddpmpp-edm-managed/network-snapshot-200000.pkl`

## 実行条件

- GPU: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`
- seeds: `0-63`
- classes: `0 1 2 3 4 5 6 7 8 9`
- steps: `18`
- batch: `64`

実行スクリプト:

```bash
bash experiments/2/scripts/generate_conditional_samples.sh
```

## 出力

- 生成画像: `experiments/2/generated/{120k,200k}/class*/`
- 実行ログ: `experiments/2/logs/generate-*.log`
- 比較用まとめ画像: `experiments/2/results/conditional_generation_120k_vs_200k.png`

## 確認結果

- 120k と 200k の両 checkpoint で `generate.py` は正常終了。
- 生成画像数は `120k/200k x 10 classes x 64 seeds = 1280` 枚。
- 比較画像を目視した限り、class 0-9 の指定に応じて CIFAR-10 の各カテゴリらしい画像が生成されている。
- 120k と 200k は同じ seed と class で比較できるように並べてある。
