# Experiment 7-5: FM/EDM Trajectory and Integration Error

## 目的

Experiment 7-3と7-4では、FMはNFEを増やすと改善する一方、同じNFE 35では
EDMよりFIDが悪かった。本実験では、この差を次の2要素に分けて調べる。

1. 有限NFEのsolverが高精度参照解から離れる**数値積分誤差**
2. 学習されたODE自体が作る**経路形状**

FIDは再計算せず、同一seed・同一class labelから得た量子化前テンソルを測定する。

## 比較方法

FMとEDMでは時間変数と初期状態の尺度が異なるため、モデル間で中間テンソルを直接
引き算しない。各モデルについて独立にRK4参照解を作り、solverの終点をその参照解と
比較する。モデル間では、無次元化した相対誤差と経路形状指標を比較する。

### 正規化進行度

- FM: `s=t`、`t: 0 -> 1`、`dx/ds = v_theta(x,s)`
- EDM: `s: 0 -> 1`をEDMのrho scheduleで`sigma: 80 -> 0.002`へ写し、
  `dx/ds = ((x-D_theta(x,sigma))/sigma) (d sigma/ds)`
- EDM終点は`D(x(0.002), 0.002)`とし、`sigma=0`でnetworkを評価しない

RK4参照解は`coarse`と`fine`の2解を作る。その終点間距離が、測定対象のsolver誤差
より十分小さいことを参照解の収束条件とする。

### solver条件

| solver | NFE | steps |
|:--|--:|--:|
| Euler | 5, 9, 18, 19, 35 | NFEと同じ |
| Heun（最終区間はEuler） | 5, 9, 19, 35 | 3, 5, 10, 18 |

FMは`generate.py`の`fm_euler_sampler` / `fm_heun_sampler`、EDM EulerはEDM
discretizationのablation sampler、EDM Heunは標準`edm_sampler`を直接呼ぶ。
したがって7-3/7-4と同じ更新式になる。

## 指標

seedごとに以下を記録する。

- `reference_relative_l2`: coarse RK4終点とfine RK4終点の相対L2
- `*_relative_l2`: 各solver終点とfine RK4終点の相対L2
- `path_length`: fine RK4軌跡の総移動距離
- `chord_length`: 初期点と終点の直線距離
- `straightness_ratio = path_length / chord_length`（1に近いほど直線的）
- `max_chord_deviation_ratio`: 直線補間経路からの最大距離 / chord length
- `mean_chord_deviation_ratio`: 上記距離の進行度平均

経路形状は状態空間と時間パラメータに依存するため、「FMが本質的に常に直線」といった
一般論ではなく、本実験のcheckpointと規約に対する診断値として扱う。

## 実行段階

| stage | seeds | batch | RK4 coarse/fine | 目的 | 予想時間 |
|:--|:--|--:|:--|:--|:--|
| pilot | `0-63` | 16 | 64 / 128 steps | 実装・参照収束・傾向確認 | 約8〜12分 |
| final | `0-255` | 16 | 128 / 256 steps | 修論用集計 | 約40〜50分 |

予想時間はGPU 0での過去実測からの概算で、GPU待機時間を含まない。

## 計画確認

```bash
PLAN_ONLY=1 \
  experiments/7/5_fm_edm_trajectory_error/scripts/schedule_after_gpu_free.sh

CONFIG=experiments/7/5_fm_edm_trajectory_error/config/final.tsv \
PLAN_ONLY=1 \
  experiments/7/5_fm_edm_trajectory_error/scripts/schedule_after_gpu_free.sh
```

## pilot実行

```bash
tmux new -s edm-exp7-5-pilot
GPU_INDEX=0 \
  experiments/7/5_fm_edm_trajectory_error/scripts/schedule_after_gpu_free.sh
```

pilot後、`reference_relative_l2`がEuler/Heun誤差より十分小さいことを確認してから
finalへ進む。

## final実行

```bash
tmux new -s edm-exp7-5-final
CONFIG=experiments/7/5_fm_edm_trajectory_error/config/final.tsv \
GPU_INDEX=0 \
  experiments/7/5_fm_edm_trajectory_error/scripts/schedule_after_gpu_free.sh
```

同じstageを再実行すると、`_SUCCESS`があればスキップする。途中停止した場合は
`per_seed_metrics.tsv`に記録済みのmodel/seedをスキップして続行する。

## 出力

```text
results/<stage>/
  run_config.json
  per_seed_metrics.tsv
  endpoint_error_stats.tsv
  path_geometry_stats.tsv
  endpoint_errors.svg
  path_geometry.svg
  README.md
  _SUCCESS
```

全軌跡テンソルやPNGは保存せず、seedごとのスカラー指標と集計図だけを残す。
