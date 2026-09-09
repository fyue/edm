# Experiment 3_8: 167526 / 165018 snapshots の3 seed評価

## 目的

実験3_3 `lr=0.0002, EMA=0.5, 160k -> 170k` の保存済みsnapshotから、次の2点をFID-50kで3 seed範囲評価します。

```text
experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/network-snapshot-167526.pkl
experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/network-snapshot-165018.pkl
```

学習は行わず、snapshotはコピーせず参照します。FID画像、ログ、summaryだけを本フォルダ内に保存します。

## 評価seed範囲

```text
0-49999
50000-99999
100000-149999
```

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/scripts/run_fid50k_167526_165018.sh
```

## 実行

```bash
tmux new -s edm-exp3-8
experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/scripts/run_fid50k_167526_165018.sh
```

tmuxから切り離す場合は `Ctrl-b` の後に `d` を押します。

## 進捗確認

```bash
tail -f experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/logs/fid50k-167526-165018.log
```

## 結果確認

```bash
cat experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/results/summary.tsv
```

## 再開

途中で停止した場合は同じスクリプトを再実行します。summaryに記録済みの評価はスキップし、画像が50,000枚そろっているseed範囲は生成をスキップしてFID計算から再開できます。

## 出力先

```text
experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/fid-evals/
experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/logs/
experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5/results/summary.tsv
```
