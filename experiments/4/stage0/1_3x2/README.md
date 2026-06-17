# Experiment 4 Stage 0 3x2: S_churn sensitivity smoke test

## 目的

CIFAR-10 class-conditional EDM の学習段階ごとに、`S_churn` と生成品質の関係を低コストに確認する。

このフォルダは、Stage 0 のうち `40k, 120k, pretrained` x `S_churn=0,20` の3x2部分実験をまとめたものである。

正式な50k FIDではなく、1024 samples の `FID-1k` を使う。これは論文値と比較するための値ではなく、条件間の相対的な傾向を見るためのスモークテストである。

## 対象

デフォルトでは次の3モデルを評価する。

| checkpoint | network |
|---|---|
| `40k` | `network-snapshot-040000.pkl` |
| `120k` | `network-snapshot-120000.pkl` |
| `pretrained` | official CIFAR-10 conditional EDM |

checkpoint一覧は `experiments/4/stage0/3x2/config/stage0_checkpoints.tsv` に置いている。設定ファイルには `10k` も残しているが、この3x2実験のデフォルト実行対象からは外している。

## Stage 0 実行条件

- `S_churn`: `0, 20`
- samples: `1024`
- seeds: `0-1023`
- steps: `18`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

## 実行方法

実行前に予定だけ確認する。

```bash
PLAN_ONLY=1 experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh
```

tmuxで実行する。

```bash
tmux new -s edm-exp4-stage0
experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh
```

進捗確認。

```bash
tail -f experiments/4/stage0/3x2/logs/stage0-churn-smoke.log
```

結果確認。

```bash
cat experiments/4/stage0/3x2/results/stage0_summary.tsv
```

FID-`S_churn` 曲線を描く。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/4/stage0/3x2/scripts/plot_stage0_churn_curve.py
```

出力:

```text
experiments/4/stage0/3x2/results/stage0_fid_vs_s_churn.svg
```

## 一部だけ実行

40k と pretrained だけ、`S_churn=0,20` だけを試す。

```bash
CHECKPOINT_FILTER="40k pretrained" CHURNS="0 20" \
  experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh
```

サンプル数を変える。

```bash
NUM_IMAGES=2048 SEEDS=0-2047 experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh
```

## 再開

同じコマンドを再実行すると、`stage0_summary.tsv` に記録済みの条件はスキップする。画像が既に十分な枚数ある場合は生成もスキップして、FID計算から再開する。

現在の条件が終わったところで止めたい場合は、別ターミナルで次を実行する。

```bash
touch experiments/4/stage0/3x2/STOP_AFTER_CURRENT_CONDITION
```

## 出力

```text
experiments/4/stage0/3x2/images/<checkpoint>/S_churn-*/n1024-seeds-0-1023/
experiments/4/stage0/3x2/logs/stage0-*.log
experiments/4/stage0/3x2/results/stage0_summary.tsv
```

## 解釈上の注意

- `FID-1k` はサンプル数が少ないため、絶対値としては不安定。
- 同じ `NUM_IMAGES`, `SEEDS`, `steps`, `S_min`, `S_max`, `S_noise` で揃えた条件間の相対比較として使う。
- Stage 0 で有望な `S_churn` が見えたら、Stage 1 で `FID-5k`、最後に重要条件だけ50k FIDで確認する。
- `fid.py calc` はInception特徴抽出にはGPUを使うが、最後の `scipy.linalg.sqrtm` によるFID値計算はCPUで実行される。そのため終盤は `nvidia-smi` のGPU利用率が低く見えても正常。
