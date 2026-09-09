# Experiment 3-11: lr=5e-4, EMA=0.5, 170k周辺4 checkpointの3 seed評価

`lr=0.0005`, `EMA=0.5` 条件で生成済みの checkpoint から、170k 近傍の4点を FID-50k の3 seed範囲で評価する。

## 対象 checkpoint

既存の Experiment 3-2 の継続学習 run を参照する。ファイルはコピーせず、FID画像・ログ・結果だけを本ディレクトリに保存する。

| label | requested kimg | snapshot kimg | network |
| --- | ---: | ---: | --- |
| 165k | 165000 | 165018 | `experiments/3/2_160k-200k-lr5e-4-ema0.5/training-runs/lr5e-4-ema0.5-160k-200k/network-snapshot-165018.pkl` |
| 167.5k | 167500 | 167526 | `experiments/3/2_160k-200k-lr5e-4-ema0.5/training-runs/lr5e-4-ema0.5-160k-200k/network-snapshot-167526.pkl` |
| 170k | 170000 | 170035 | `experiments/3/2_160k-200k-lr5e-4-ema0.5/training-runs/lr5e-4-ema0.5-160k-200k/network-snapshot-170035.pkl` |
| 172.5k | 172500 | 172544 | `experiments/3/2_160k-200k-lr5e-4-ema0.5/training-runs/lr5e-4-ema0.5-160k-200k/network-snapshot-172544.pkl` |

この4点は `config/checkpoints.tsv` で管理する。

## 評価設定

```text
FID seeds = 0-49999, 50000-99999, 100000-149999
num images = 50000
sampler steps = 18
S_churn = 0
S_min = 0
S_max = inf
S_noise = 1
```

## 実行

事前確認:

```bash
PLAN_ONLY=1 experiments/3/11_fid50k_around170k_lr5e-4_ema0.5/scripts/run_fid50k_around170k.sh
```

本実行:

```bash
tmux new -s edm-exp3-11
experiments/3/11_fid50k_around170k_lr5e-4_ema0.5/scripts/run_fid50k_around170k.sh
```

ログ確認:

```bash
tail -f experiments/3/11_fid50k_around170k_lr5e-4_ema0.5/logs/fid50k-around170k.log
```

結果:

```bash
cat experiments/3/11_fid50k_around170k_lr5e-4_ema0.5/results/summary.tsv
cat experiments/3/11_fid50k_around170k_lr5e-4_ema0.5/results/aggregate.tsv
```

`summary.tsv` は checkpoint x seed range の生結果、`aggregate.tsv` は checkpointごとの3 seed平均・標準偏差・最小・最大を記録する。

## 中断

現在の checkpoint の3 seed評価が終わったところで止めたい場合:

```bash
touch experiments/3/11_fid50k_around170k_lr5e-4_ema0.5/STOP_AFTER_CURRENT_CHECKPOINT
```
