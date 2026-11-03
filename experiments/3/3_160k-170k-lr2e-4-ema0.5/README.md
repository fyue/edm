# Experiment 3 continuation: lr=0.0002, EMA=0.5, 160k -> 170k

3-1のlr=5e-4の160kから、lr=2e-4に下げて170kまで継続する。170kのFID-50k（3 seed平均）は1.939で、起点の2.010より改善した。

## 目的

`120k-160k-lr-ema-sweep` で最も良いFIDを示した 160k checkpoint から、
EMA half-lifeは `0.5` のまま learning rate を `0.0002` に下げて 170k まで継続します。
170k時点のFIDを確認するための実験です。

## 起点

```text
experiments/3/1_120k-160k-lr-ema-sweep/training-runs/lr5e-4-ema0.5/training-state-160000.pt
experiments/3/1_120k-160k-lr-ema-sweep/training-runs/lr5e-4-ema0.5/network-snapshot-160000.pkl
```

## 設定

```text
lr = 0.0002
EMA half-life = 0.5
start = 160000 kimg
target = 170000 kimg
eval = 170000 kimg
```

FIDはデフォルトで3つの50k seed範囲を評価します。

```text
0-49999
50000-99999
100000-149999
```

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/3_160k-170k-lr2e-4-ema0.5/scripts/run_160k_to_170k_lr2e-4_ema0.5.sh
```

## 実行

```bash
tmux new -s edm-exp3-170k
experiments/3/3_160k-170k-lr2e-4-ema0.5/scripts/run_160k_to_170k_lr2e-4_ema0.5.sh
```

## 進捗と結果の確認

```bash
tail -f experiments/3/3_160k-170k-lr2e-4-ema0.5/logs/train-eval.log
cat experiments/3/3_160k-170k-lr2e-4-ema0.5/results/summary.tsv
cat experiments/3/3_160k-170k-lr2e-4-ema0.5/results/health.tsv
```

途中で停止した場合は同じコマンドを再実行すると、この実験フォルダ内の
最新 `training-state-*.pt` から再開します。
