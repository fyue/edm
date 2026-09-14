# Experiment 8-2: Selected Restart K FID-5k Confirmation

## 目的

Experiment 8-1のFID-1k screeningで観測した「学習不足modelほどRestartの改善幅が
大きい」という傾向を、FID-5kで再確認する。

まず`K=5`を全checkpointで固定して学習進行に対する効果を確認する。さらに8-1で
改善した`K=1,2`を5.5k、20kで評価し、少ないNFEで改善を維持できるかとKに対する
非単調性を確認する。

## 実験条件

- dataset/model: CIFAR-10 class-conditional EDM
- samples: 5,000/条件
- seeds: `0-4999`（全条件で共通）
- Restart: main 18 steps、`N_restart=3`、`t_min=0.06`、`t_max=0.30`
- Restart noise multiplier: `1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- GPU: 1

全checkpointで次のK=5比較を行う。

| 条件 | main steps | K | NFE | 目的 |
|---|---:|---:|---:|---|
| ODE main baseline | 18 | 0 | 35 | 同じmain scheduleとの比較 |
| Restart | 18 | 5 | 55 | 評価対象 |
| ODE equal-NFE baseline | 28 | 0 | 55 | 同じ計算量との比較 |

checkpointは`5.5k`、`20k`、`120k`、`200k`の4点で、K=5 subsetは
12条件・6万画像。
さらに5.5kと20kには次を追加する。

| 追加条件 | main steps | K | NFE | 比較対象 |
|---|---:|---:|---:|---|
| Restart K=1 | 18 | 1 | 39 | 20-step ODE（39 NFE） |
| Restart K=2 | 18 | 2 | 43 | 22-step ODE（43 NFE） |

合計は20条件・10万画像。条件の詳細は
[config/restart_k5_fid5k.tsv](config/restart_k5_fid5k.tsv)に固定する。directory名と
config filenameの`k5`は最初の12条件をK=5で開始した履歴を保持するため変更しない。

## 設計理由

8-1の同NFE比較におけるK=5のFID-1k改善幅は、5.5kで`+2.2140`、20kで
`+0.5627`、120kで`+0.0718`、200kで`-0.0634`だった。K=5を全checkpointで
固定する比較に加え、5.5kと20kではK=1,2も正の改善だったため、FID-5kで
品質対NFEのtrade-offを確認する。

FID-5kはscreeningより安定した確認値だが、最終的な主張には使わない。傾向が再現した
場合、Experiment 8-3で同じ固定条件をFID-50k × 3 seed範囲へ進める。

## 検証

GPUを使わず、全20条件、checkpoint path、NFEを確認する。

```bash
PLAN_ONLY=1 \
  experiments/8/2_restart_k5_fid5k_confirmation/scripts/schedule_after_gpu_free.sh
```

必要なら実checkpointで2枚だけ生成する。

```bash
SAMPLES_ONLY=1 SAMPLE_SEEDS=0-1 CONDITION_FILTER="5p5k-restart-k5" GPU_INDEX=0 \
  experiments/8/2_restart_k5_fid5k_confirmation/scripts/schedule_after_gpu_free.sh
```

## 実行

GPU 0が空くまで共通wrapperで待ってから全条件を実行する。

```bash
GPU_INDEX=0 \
  experiments/8/2_restart_k5_fid5k_confirmation/scripts/schedule_after_gpu_free.sh
```

学習不足checkpointだけ先行する場合:

```bash
CHECKPOINT_FILTER="5.5k 20k" GPU_INDEX=0 \
  experiments/8/2_restart_k5_fid5k_confirmation/scripts/schedule_after_gpu_free.sh
```

進捗確認:

```bash
tail -f experiments/8/2_restart_k5_fid5k_confirmation/logs/restart-k5-fid5k.log
```

同一条件・画像数・seed範囲・FID referenceが結果TSVに記録済みなら再実行時にskipする。
画像が5,000枚揃っている場合は生成をskipしてFID計算から再開する。

現在の条件終了後に安全停止するには次を作成する。

```bash
touch experiments/8/2_restart_k5_fid5k_confirmation/STOP_AFTER_CURRENT_CONDITION
```

再開前に停止fileを削除する。

## 集計とグラフ

完走時にrunnerが自動生成する。手動で再生成する場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/summarize_results.py \
  --input experiments/8/2_restart_k5_fid5k_confirmation/results/restart_fid_summary.tsv \
  --output experiments/8/2_restart_k5_fid5k_confirmation/results/restart_improvements.tsv

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/2_restart_k5_fid5k_confirmation/scripts/plot_results.py
```

正の改善幅はRestartのFIDがbaselineより低いことを表す。特に
`improvement_vs_same_nfe`を主比較とする。再実行時、既に完了したK=5とbaselineの
12条件はskipされ、新規K=1,2と同NFE baselineの8条件だけを実行する。

## 出力

```text
experiments/8/2_restart_k5_fid5k_confirmation/
  images/<condition>/fid-n5000-seeds-0-4999/
  logs/
  results/restart_fid_summary.tsv
  results/restart_improvements.tsv
  results/restart_fid5k_improvements.{png,svg}
```

生成画像とlogはcommitしない。
