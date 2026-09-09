# Experiment 3-10: 160k FID~2.05 checkpoint -> 170k, lr=2e-4

## 目的

Experiment 3-1 の `lr1e-3-ema0.5-repro` 分岐で得られた、160k時点の3 seed平均FIDがおよそ `2.05` のcheckpointを起点に、`lr=0.0002`, `EMA=0.5` で170kまで継続学習する。

学習が170kまで完了した後、160kより後に生成された各 `network-snapshot-*.pkl` を、それぞれ3つのFID-50k seed範囲で評価する。

## 起点

```text
experiments/3/1_120k-160k-lr-ema-sweep/training-runs/lr1e-3-ema0.5-repro/training-state-160000.pt
experiments/3/1_120k-160k-lr-ema-sweep/training-runs/lr1e-3-ema0.5-repro/network-snapshot-160000.pkl
```

160k時点の既存FID:

```text
0-49999        2.06615
50000-99999    2.04264
100000-149999  2.04078
mean           2.04986
```

## 設定

```text
start = 160000 kimg
target = 170000 kimg
lr = 0.0002
EMA half-life = 0.5
FID seeds = 0-49999, 50000-99999, 100000-149999
FID images = 50000
sampling steps = 18
S_churn = 0
```

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/scripts/run_160k_fid2.05_to_170k_lr2e-4_then_eval_all.sh
```

## 実行

```bash
tmux new -s edm-exp3-10
experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/scripts/run_160k_fid2.05_to_170k_lr2e-4_then_eval_all.sh
```

## 進捗確認

```bash
tail -f experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/logs/train-then-eval-all.log
```

## 結果

```bash
cat experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/results/summary.tsv
cat experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/results/aggregate.tsv
cat experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/results/health.tsv
```

`summary.tsv` は checkpoint x seed range の生結果、`aggregate.tsv` は checkpointごとの3 seed平均・標準偏差・最小・最大を記録する。

## 再開と停止

同じコマンドを再実行すると、既存の最新 `training-state-*.pt` から学習を再開し、記録済みFID条件はスキップする。

現在のcheckpoint評価が終わったところで止める:

```bash
touch experiments/3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/STOP_AFTER_CURRENT_CHECKPOINT
```
