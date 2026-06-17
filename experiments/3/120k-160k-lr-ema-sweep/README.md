# Experiment 3: 120kからのlr/EMA分岐実験

## 目的

Experiment 1では、CIFAR-10 class-conditional DDPM++ EDM の本番学習で 120k のFIDが最も良く、その後 160k/200k で悪化しました。

この実験では 120k checkpoint を起点にして、120k -> 160k の区間だけ lr と EMA half-life を変え、FIDの挙動を確認します。

比較する条件は次の3つです。

| branch_id | lr | ema | 目的 |
|---|---:|---:|---|
| `lr1e-3-ema0.5-repro` | 0.001 | 0.5 | 元設定の再現確認 |
| `lr5e-4-ema0.5` | 0.0005 | 0.5 | 120k以降の学習率を下げる |
| `lr1e-3-ema1.0` | 0.001 | 1.0 | EMA half-lifeを長くする |

## 起点ファイル

デフォルトでは次の120kファイルを参照します。

```bash
experiments/1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/training-runs/cifar10-cond-ddpmpp-edm-managed/training-state-120000.pt
experiments/1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/training-runs/cifar10-cond-ddpmpp-edm-managed/network-snapshot-120000.pkl
```

`train.py --resume` は `training-state-120000.pt` を読み、その隣の `network-snapshot-120000.pkl` も使います。

## 実行方法

tmux内で実行します。

```bash
tmux new -s edm-exp3
experiments/3/scripts/run_lr_ema_branches_from_120k.sh
```

tmuxから切り離す場合:

```text
Ctrl-b を押してから d
```

tmuxに戻る場合:

```bash
tmux attach -t edm-exp3
```

## 進捗確認

```bash
tail -f experiments/3/logs/lr-ema-branches.log
```

分岐ごとの学習ログ:

```bash
tail -f experiments/3/training-runs/lr1e-3-ema0.5-repro/log.txt
tail -f experiments/3/training-runs/lr5e-4-ema0.5/log.txt
tail -f experiments/3/training-runs/lr1e-3-ema1.0/log.txt
```

## 結果確認

```bash
cat experiments/3/results/summary.tsv
cat experiments/3/results/health.tsv
```

`summary.tsv` には、分岐条件、seed範囲、FID、評価snapshot、生成画像ディレクトリが記録されます。

## 再開方法

途中で止まった場合は、同じコマンドを再実行します。

```bash
experiments/3/scripts/run_lr_ema_branches_from_120k.sh
```

各分岐に `training-state-*.pt` があれば、その最新stateから再開します。
該当分岐にstateがまだ無い場合は、120k起点から始め直します。
FIDが `summary.tsv` に記録済みのseed範囲はスキップします。

## 分岐単位で安全に止める方法

現在の分岐が完了したところで止めたい場合:

```bash
touch experiments/3/STOP_AFTER_CURRENT_BRANCH
```

## 出力先

```text
experiments/3/training-runs/<branch_id>/
experiments/3/fid-evals/<branch_id>/
experiments/3/results/summary.tsv
experiments/3/results/health.tsv
experiments/3/logs/lr-ema-branches.log
```

## 注意

- 3分岐それぞれで 120k -> 160k の40,000 kimgを学習します。
- 各分岐で 50,000枚 x 3 seed範囲のFID評価を行うため、評価だけでもかなり時間がかかります。
- GPUは `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` で1基のみ使用します。
- datasetとFID refは既存の `datasets/cifar10-32x32.zip` と `fid-refs/cifar10-32x32.npz` を使います。
