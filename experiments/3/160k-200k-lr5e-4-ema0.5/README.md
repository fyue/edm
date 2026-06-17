# Experiment 3 continuation: lr=0.0005, EMA=0.5, 160k -> 200k

## 目的

`120k-160k-lr-ema-sweep` では、120k -> 160k の分岐実験で `lr=0.0005, EMA=0.5` が最も良いFIDを示しました。

この実験では、その 160k checkpoint を起点として、同じ `lr=0.0005, EMA=0.5` のまま 200k まで継続し、160k以降もFID悪化を抑えられるかを確認します。

## 起点

```text
experiments/3/120k-160k-lr-ema-sweep/training-runs/lr5e-4-ema0.5/training-state-160000.pt
experiments/3/120k-160k-lr-ema-sweep/training-runs/lr5e-4-ema0.5/network-snapshot-160000.pkl
```

## 設定

```text
lr = 0.0005
EMA half-life = 0.5
start = 160000 kimg
target = 200000 kimg
eval = 180000, 200000 kimg
```

デフォルトでは、180kは `0-49999` の1 seed範囲で確認し、200kは3 seed範囲で評価します。

```text
180k: 0-49999
200k: 0-49999, 50000-99999, 100000-149999
```

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh
```

## 実行

tmux内で実行します。

```bash
tmux new -s edm-exp3-200k
experiments/3/160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh
```

## 進捗確認

```bash
tail -f experiments/3/160k-200k-lr5e-4-ema0.5/logs/train-eval.log
```

## 結果確認

```bash
cat experiments/3/160k-200k-lr5e-4-ema0.5/results/summary.tsv
cat experiments/3/160k-200k-lr5e-4-ema0.5/results/health.tsv
```

## 再開

途中で止まった場合は同じコマンドを再実行します。新しい実験フォルダ内の最新 `training-state-*.pt` から再開します。

## 安全に止める

現在の 180k または 200k ステージが終わったところで止めたい場合:

```bash
touch experiments/3/160k-200k-lr5e-4-ema0.5/STOP_AFTER_CURRENT_STAGE
```
