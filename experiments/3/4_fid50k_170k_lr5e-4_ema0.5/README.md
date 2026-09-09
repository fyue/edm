# 実験3-4: lr=0.0005, EMA=0.5 の170k FID-50k

## 目的

`160k-200k-lr5e-4-ema0.5` の学習途中に保存された170k checkpointについて、
FID-50kを計算します。学習は行わず、保存済みsnapshotから画像生成とFID計算だけを実行します。

## 対象checkpoint

```text
experiments/3/160k-200k-lr5e-4-ema0.5/training-runs/lr5e-4-ema0.5-160k-200k/network-snapshot-170035.pkl
```

保存周期の都合でファイル名は `170035` ですが、本実験では170k checkpointとして扱います。

## デフォルト設定

```text
learning rate = 0.0005
EMA half-life = 0.5
images = 50000
seed ranges = 0-49999, 50000-99999, 100000-149999
steps = 18
S_churn = 0
```

既存評価で使った `0-49999` に2つのseed範囲を追加し、合計3回のFID-50kを評価します。
各seed範囲の画像とFIDは独立して保存・記録されます。

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/4_fid50k_170k_lr5e-4_ema0.5/scripts/run_fid50k_170k.sh
```

## 実行

```bash
tmux new -s edm-exp3-4-fid170k
experiments/3/4_fid50k_170k_lr5e-4_ema0.5/scripts/run_fid50k_170k.sh
```

## 進捗・結果確認

```bash
tail -f experiments/3/4_fid50k_170k_lr5e-4_ema0.5/logs/fid50k-170k.log
cat experiments/3/4_fid50k_170k_lr5e-4_ema0.5/results/summary.tsv
```

途中で停止した場合は同じコマンドを再実行できます。各seed範囲について、50,000枚が揃っていれば
画像生成を省略し、結果が記録済みならそのseed範囲を省略します。
