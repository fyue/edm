# Experiment 3_6: lr=0.0001, EMA=0.5, 170k -> 180k

## 目的

実験3_3で改善した170k checkpointを起点に、learning rateを `0.0001` に下げて
175k、180kまで段階的に継続して評価します。加えて、起点の170kも同じ条件で
再評価し、ベースラインの再現性を確認します。実験3_5（`lr=0.0002`）との比較実験です。

## 起点

```text
experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/training-state-170000.pt
experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k/network-snapshot-170000.pkl
```

起点ファイルはコピーせず参照します。170k評価には上記snapshotを直接使用し、
175kと180kのstate、snapshot、および全評価のFID画像は本フォルダ内に保存します。

## 実験設定

```text
lr = 0.0001
EMA half-life = 0.5
start = 170000 kimg
target = 180000 kimg
eval = 170000, 175000, 180000 kimg
```

170k、175k、180kの各時点で次の3 seed範囲を評価します。

```text
0-49999
50000-99999
100000-149999
```

## 実行前確認

```bash
PLAN_ONLY=1 experiments/3/6_170k-180k-lr1e-4-ema0.5/scripts/run_170k_to_180k_lr1e-4_ema0.5.sh
```

## 実行

```bash
tmux new -s edm-exp3-6
experiments/3/6_170k-180k-lr1e-4-ema0.5/scripts/run_170k_to_180k_lr1e-4_ema0.5.sh
```

tmuxから切り離す場合は `Ctrl-b` の後に `d` を押します。

## 進捗と結果の確認

```bash
tail -f experiments/3/6_170k-180k-lr1e-4-ema0.5/logs/train-eval.log
cat experiments/3/6_170k-180k-lr1e-4-ema0.5/results/summary.tsv
cat experiments/3/6_170k-180k-lr1e-4-ema0.5/results/health.tsv
```

途中で停止した場合は同じコマンドを再実行します。実験3_6内の最新
`training-state-*.pt` から再開し、記録済みのFID評価はスキップします。
