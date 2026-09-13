# Experiment 7-4: FM/EDM FID-50k Confirmation

## 目的

Experiment 7-3のFID-5kで得た主要傾向を、標準的な50,000生成画像のFIDで確認する。
20,000 kimgまで学習した同一規模のCIFAR-10 conditionalモデルを使い、次の2点を検証する。

1. FM EulerでNFEを18から35へ増やしたとき、生成品質の改善がFID-50kでも残るか。
2. 同一NFE 35で、7-3における最良FM条件と標準EDM条件の差がFID-50kでも残るか。

この実験は学習を行わず、保存済みcheckpointの評価だけを行う。

## 評価条件

| condition | model / sampler | steps | NFE | FID-5k |
|:--|:--|--:|--:|--:|
| `fm-euler-nfe18` | FM Euler | 18 | 18 | 11.7926 |
| `fm-euler-nfe35` | FM Euler | 35 | 35 | 9.86406 |
| `edm-heun-nfe35` | standard EDM Heun | 18 | 35 | 7.6149 |

全条件で生成seedを`0-49999`、FID参照統計を`fid-refs/cifar10-32x32.npz`に固定する。
checkpoint、既存FID-5k値、既存画像の場所は
[config/conditions.tsv](config/conditions.tsv)に記録している。

EDM Heun・NFE 35は、Experiment 4で同じcheckpoint、sampler、steps、`S_churn=0`、
seed範囲を使って生成した50,000枚を再利用する。既存値はFID-50k `2.98971`だが、
7-4の結果表に同じ手順で記録するためFIDを再計算する。FMの2条件は新規生成する。

FM EulerとEDM Heunの比較は「同じsolverの比較」ではなく、同じNFE予算における
7-3の最良FM条件と標準EDM条件の比較である。同一solverでの比較は7-3のFID-5k結果を
参照する。

## 仮説と読み方

- `FM Euler NFE 35 < FM Euler NFE 18`なら、FMのNFE増加による改善を確認できる。
- `EDM Heun NFE 35 < FM Euler NFE 35`なら、7-3で見えたEDM優位は5,000標本だけの
  揺れでは説明しにくい。
- FID-5kとFID-50kの絶対値は直接比較せず、同じ画像数の条件間比較に使う。
- FID-50kはseed範囲を変えた反復実験ではないため、seed間分散の推定は行わない。

## 実行

GPUを使わず、3条件、checkpoint、既存EDM画像、NFE、seed範囲を検証する。

```bash
PLAN_ONLY=1 \
  experiments/7/4_fm_edm_fid50k_confirmation/scripts/schedule_after_gpu_free.sh
```

GPU 0が空くまで待ってから全条件を実行する。

```bash
tmux new -s edm-exp7-4
GPU_INDEX=0 \
  experiments/7/4_fm_edm_fid50k_confirmation/scripts/schedule_after_gpu_free.sh
```

一部の条件だけを実行する場合は空白区切りで指定する。

```bash
CONDITION_FILTER="fm-euler-nfe18" GPU_INDEX=0 \
  experiments/7/4_fm_edm_fid50k_confirmation/scripts/schedule_after_gpu_free.sh
```

現在の条件が終わったところで安全停止するには、次のファイルを作る。

```bash
touch experiments/7/4_fm_edm_fid50k_confirmation/STOP_AFTER_CURRENT_CONDITION
```

結果TSVに記録済みの条件は再実行時にスキップする。生成途中で停止したFM条件は、
同じseed範囲を決定論的に再生成してからFID計算を行う。

## 出力と所要時間

```text
experiments/7/4_fm_edm_fid50k_confirmation/
  images/<condition>/fid-n50000-seeds-0-49999/
  logs/fid50k-confirmation.log
  results/fid50k_confirmation.tsv
```

7-3の実測を50,000枚へ換算すると、FM 18 NFEの生成は約37分、FM 35 NFEは約72分。
FID計算と起動時間を含む全体は、GPU待機時間を除いて約2〜2.5時間を見込む。
新規画像は100,000枚で、追加ディスク使用量は約400〜500 MBの見込みである。
