# Experiment 3_5: lr=0.0002, EMA=0.5, 170k -> 180k + 175k再評価

3-3の170kから同じlr=2e-4で継続する（200kの予定を180kで打ち切った）。FID-50k（3 seed平均）は175kで2.274、180kで2.298と、170kの1.939から悪化した。

## 目的

実験3_3では、`lr=0.0002, EMA=0.5` で160kから170kまで継続し、170k時点でFIDが改善しました。

```text
seed 0-49999:       1.96391
seed 50000-99999:   1.92450
seed 100000-149999: 1.92967
3 seed平均:         1.93936
```

実験3_5では、この170k checkpointを起点に同じ設定で学習しました。当初は200kまで継続する計画でしたが、180kの3 seed評価完了後に学習を終了し、保存済みの約175k checkpointを3 seedで評価する方針へ変更しました。

## 起点

```text
experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/training-state-170000.pt
experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/network-snapshot-170000.pkl
```

起点ファイルはコピーせず参照します。実験3_5の新しいstate、snapshot、FID画像だけを本フォルダ内に保存します。

## 実験設定

```text
lr = 0.0002
EMA half-life = 0.5
start = 170000 kimg
training end = 180000 kimg
evaluation = 180000 kimg and snapshot 175018 (~175k)
```

180kと約175kの各時点で次の3 seed範囲を評価します。

```text
0-49999
50000-99999
100000-149999
```

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/5_170k-200k-lr2e-4-ema0.5/scripts/run_170k_to_200k_lr2e-4_ema0.5.sh
```

## 実行

```bash
tmux new -s edm-exp3-5
experiments/3/5_170k-200k-lr2e-4-ema0.5/scripts/run_170k_to_200k_lr2e-4_ema0.5.sh
```

180k完了・停止後の約175k評価:

```bash
experiments/3/5_170k-200k-lr2e-4-ema0.5/scripts/run_fid50k_175k.sh
```

今回の方針変更では `STOP_AFTER_CURRENT_STAGE` を置き、`handoff_180k_to_175k_fid.sh` が180kの3番目のseed評価と停止完了を待ってから、約175kの3 seed評価を開始します。対象ファイルは保存周期上の `network-snapshot-175018.pkl` です。

tmuxから切り離す場合は `Ctrl-b` の後に `d` を押します。

## 進捗確認

```bash
tail -f experiments/3/5_170k-200k-lr2e-4-ema0.5/logs/train-eval.log
```

## 結果確認

```bash
cat experiments/3/5_170k-200k-lr2e-4-ema0.5/results/summary.tsv
cat experiments/3/5_170k-200k-lr2e-4-ema0.5/results/health.tsv
```

## 再開

途中で停止した場合は各スクリプトを再実行できます。学習側と約175k評価側はいずれも記録済みのFID評価をスキップします。

## ステージ単位で停止

現在の180k、190k、または200kステージが完了したところで停止する場合:

```bash
touch experiments/3/5_170k-200k-lr2e-4-ema0.5/STOP_AFTER_CURRENT_STAGE
```

## 出力先

```text
experiments/3/5_170k-200k-lr2e-4-ema0.5/training-runs/
experiments/3/5_170k-200k-lr2e-4-ema0.5/fid-evals/
experiments/3/5_170k-200k-lr2e-4-ema0.5/results/summary.tsv
experiments/3/5_170k-200k-lr2e-4-ema0.5/results/health.tsv
experiments/3/5_170k-200k-lr2e-4-ema0.5/logs/train-eval.log
```
