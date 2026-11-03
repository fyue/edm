# Experiment 3-9: lr=2e-4 / lr=1e-4 の 172509 / 175018 snapshot 3 seed FID評価

3-5（lr=2e-4）と3-6（lr=1e-4）の172.5k・175kをFID-50k×3 seed範囲で評価する（学習なし）。3 seed平均は2.254〜2.302で、どれも170kより悪い。

## 目的

次の4条件について、FID-50kを3 seed範囲で評価します。学習は行わず、保存済みsnapshotを参照します。

```text
lr=2e-4, network-snapshot-172509.pkl
lr=2e-4, network-snapshot-175018.pkl
lr=1e-4, network-snapshot-172509.pkl
lr=1e-4, network-snapshot-175000.pkl
```

評価条件は `config/conditions.tsv` に列挙しています。

## 評価seed範囲

```text
0-49999
50000-99999
100000-149999
```

合計は4条件 x 3 seed範囲 = 12 FID評価です。

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/scripts/run_fid50k_172509_175018_lr2e-4_lr1e-4.sh
```

## 実行

```bash
tmux new -s edm-exp3-9
experiments/3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/scripts/run_fid50k_172509_175018_lr2e-4_lr1e-4.sh
```

tmuxから切り離す場合は `Ctrl-b` の後に `d` を押します。

## 進捗確認

```bash
tail -f experiments/3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/logs/fid50k-172509-175018-lr2e-4-lr1e-4.log
```

## 結果確認

```bash
cat experiments/3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/results/summary.tsv
cat experiments/3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/results/aggregate.tsv
```

## 再開

途中で停止した場合は同じスクリプトを再実行します。summaryに記録済みの評価はスキップし、画像が50,000枚そろっているseed範囲は生成をスキップしてFID計算から再開できます。

## 注意

`lr=2e-4` 側は `network-snapshot-175018.pkl`、`lr=1e-4` 側は保存済みの `network-snapshot-175000.pkl` を使います。
