# Experiment 3-13: best 170k checkpoint -> 180k, lr=1e-4

3-10の最良170k（1.851）からlr=1e-4で180kまで継続する。172.5k〜180kのFID-50k（3 seed平均）は2.026〜2.135で、すべて170kより悪化した。

## 目的

Experiment 3-10 で得られた最良の 170k checkpoint を起点に、`lr=0.0001`, `EMA=0.5` で180kまで継続学習する。

学習が180kまで完了した後、170kより後に生成された各 `network-snapshot-*.pkl` を、それぞれ3つのFID-50k seed範囲で評価する。

Experiment 3-10 の170k checkpoint評価。この 170k が既存の170k候補の中で最良:

```text
snapshot 170000
seed 0-49999        1.87224
seed 50000-99999    1.82607
seed 100000-149999  1.8554
mean                1.85124
```

## 起点

```text
experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/training-state-170000.pt
experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/network-snapshot-170000.pkl
```

起点ファイルはコピーせず参照する。3-13 の新しい state、snapshot、FID画像、ログ、結果だけを本ディレクトリに保存する。

## 設定

```text
start = 170000 kimg
target = 180000 kimg
lr = 0.0001
EMA half-life = 0.5
FID seeds = 0-49999, 50000-99999, 100000-149999
FID images = 50000
sampling steps = 18
S_churn = 0
```

`TICK_KIMG=50`, `SNAP_TICKS=50`, `DUMP_TICKS=50` なので、保存周期に従って途中 checkpoint が生成される。

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/13_170k-to-180k-lr1e-4-ema0.5/scripts/run_170k_to_180k_lr1e-4_then_eval_all.sh
```

## 実行

```bash
tmux new -s edm-exp3-13
experiments/3/13_170k-to-180k-lr1e-4-ema0.5/scripts/run_170k_to_180k_lr1e-4_then_eval_all.sh
```

## 進捗確認

```bash
tail -f experiments/3/13_170k-to-180k-lr1e-4-ema0.5/logs/train-then-eval-all.log
```

## 結果

```bash
cat experiments/3/13_170k-to-180k-lr1e-4-ema0.5/results/summary.tsv
cat experiments/3/13_170k-to-180k-lr1e-4-ema0.5/results/aggregate.tsv
cat experiments/3/13_170k-to-180k-lr1e-4-ema0.5/results/health.tsv
```

`summary.tsv` は checkpoint x seed range の生結果、`aggregate.tsv` は checkpointごとの3 seed平均・標準偏差・最小・最大を記録する。

## 再開と停止

同じコマンドを再実行すると、既存の最新 `training-state-*.pt` から学習を再開し、記録済みFID条件はスキップする。

現在のcheckpoint評価が終わったところで止める:

```bash
touch experiments/3/13_170k-to-180k-lr1e-4-ema0.5/STOP_AFTER_CURRENT_CHECKPOINT
```
