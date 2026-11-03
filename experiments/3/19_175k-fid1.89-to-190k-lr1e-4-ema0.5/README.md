# Experiment 3-19: 175k FID 1.89 checkpoint -> 190k, lr=1e-4

3-18の175k（1.890）から同じlr=1e-4で190kまで継続する。FID-50k（3 seed平均）の最良は185kの1.910で、190kでは2.027と悪化した。

## 目的

Experiment 3-18 の 175k checkpoint のうち、3 seed FID平均が約 `1.89` の枝を起点に、3-18 と同じ `lr=0.0001`, `EMA=0.5` で 190k まで継続学習する。

学習が 190k まで完了した後、175k より後に生成された各 `network-snapshot-*.pkl` を、それぞれ3つの FID-50k seed 範囲で評価する。

## 起点

```text
experiments/3/18_160k-fid2.05-to-175k-lr1e-4-ema0.5/training-runs/lr1e-4-ema0.5-160k-175k/training-state-175000.pt
experiments/3/18_160k-fid2.05-to-175k-lr1e-4-ema0.5/training-runs/lr1e-4-ema0.5-160k-175k/network-snapshot-175000.pkl
```

起点ファイルはコピーせず参照する。3-19 の新しい state、snapshot、FID画像、ログ、結果だけを本ディレクトリに保存する。

## 設定

```text
start = 175000 kimg
target = 190000 kimg
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
PLAN_ONLY=1 experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/scripts/run_175k_fid1.89_to_190k_lr1e-4_then_eval_all.sh
```

## 既存実験完了後に実行

```bash
tmux new -s edm-exp3-19
experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/scripts/schedule_after_existing_experiments.sh
```

このランチャーは、既存の EDM 実験コマンドが動いている間は待機し、消えたら 3-19 の本体を開始する。

## 直接実行

```bash
tmux new -s edm-exp3-19
experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/scripts/run_175k_fid1.89_to_190k_lr1e-4_then_eval_all.sh
```

## 進捗確認

```bash
tail -f experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/logs/schedule-after-existing.log
tail -f experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/logs/train-then-eval-all.log
```

## 結果

```bash
cat experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/results/summary.tsv
cat experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/results/aggregate.tsv
cat experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/results/health.tsv
```

`summary.tsv` は checkpoint x seed range の生結果、`aggregate.tsv` は checkpoint ごとの3 seed平均・標準偏差・最小・最大を記録する。

## 再開と停止

同じコマンドを再実行すると、既存の最新 `training-state-*.pt` から学習を再開し、記録済み FID 条件はスキップする。

現在の checkpoint 評価が終わったところで止める:

```bash
touch experiments/3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/STOP_AFTER_CURRENT_CHECKPOINT
```
