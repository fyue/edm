# Experiment 5-13: Trajectory Distance for 5-4 vs 5-11

## Purpose

実験5-4（不等間隔に合わせた可変係数AB）と実験5-11（等間隔用の固定係数
PNDM no-RK）が、同じ学習済みdenoiserのODEに対する高精度参照軌道からどれだけ
離れるかを測る。

確認したい仮説は次である。

```text
5-4:  ODE参照解には近いがFIDは悪い
5-11: ODE参照解からは遠いがFIDは良い
```

参照した手順:

```text
/home/kyoukinkin/data/projects/lecture_handouts/個人メモ/技術知識まとめ/
拡散モデルにおける軌道距離の測り方（EDM・AB係数比較）.md
```

## Measurements

同一seedから一度だけlatentとclass labelを作り、すべての手法で共有する。PNGは
生成せず、量子化前のテンソルからseedごとの相対L2距離を計算する。

- RK4各区間4分割と8分割の距離（参照解の収束確認）
- 5-4、5-11の大域誤差
- 参照状態と参照微分履歴へ各AB係数だけを適用した局所1-step誤差
- 各sigmaにおけるdenoised状態距離
- stepごとのmean、median、sample standard deviation、90 percentile
- 横軸log sigma、縦軸relative L2のSVGグラフ

参照RK4は`80 → 0.002`まで積分する。`sigma=0`ではODEを評価せず、参照最終画像を
`D(x_ref(0.002), 0.002)`とする。全画像テンソルは保存せず、seedごとの距離だけを
`per_seed_metrics.tsv`へ追記するため、中断後も完了済みseedから再開できる。

## Conditions

| stage | seeds | batch | steps | RK4 subdivisions | purpose |
|---|---:|---:|---:|---|---|
| pilot | `0-255` | 64 | 18 | 4, 8 | 動作と傾向、参照収束の確認 |
| final | `0-1023` | 64 | 18 | 4, 8 | 最終集計 |

共通条件:

- checkpoint: EDM CIFAR-10 class-conditional pretrained
- sigma schedule: EDM rho schedule (`sigma_max=80`, `sigma_min=0.002`, `rho=7`)
- 5-4: variable-coefficient AB1 → AB2 → AB3 → AB4
- 5-11: fixed-coefficient AB1 → AB2 → AB3 → AB4, no RK warmup
- `S_churn=0`
- physical GPU index: `0`

## Expected Runtime

1画像あたりのnetwork評価数は次のとおり。

| component | NFE/image |
|---|---:|
| RK4 4 subdivisions: `17 × 4 × 4 + terminal denoise` | 273 |
| RK4 8 subdivisions: `17 × 8 × 4 + terminal denoise` | 545 |
| 5-4 variable AB | 18 |
| 5-11 fixed AB | 18 |
| total | **854** |

denoised距離と局所誤差には、各軌道の計算中に得たdenoiser出力と微分を再利用するため、
追加NFEは発生しない。

実験5-11と5-12の実測生成時間から、GPU 0（Quadro RTX 8000）では
約`0.00244 sec / (NFE × image)`と見積もる。

| stage | theoretical estimate | operational estimate |
|---|---:|---:|
| pilot 256 seeds | 約8.9分 | **10–15分** |
| final 1024 seeds | 約35.6分 | **40–50分** |
| pilot + final | 約44.5分 | **50–65分** |

finalは独立した1024-seed集計にするため、pilotの256 seedsも再計算する。上記にはGPUが
空くまでの待ち時間、初回network download、予期しない再実行時間は含まない。

## Plan Check

pilot条件と時間見積もりをGPU待機なしで確認する。

```bash
PLAN_ONLY=1 \
  experiments/5/13_trajectory_distance_5-4_vs_5-11/scripts/run_trajectory_distance.sh
```

final条件を確認する。

```bash
PLAN_ONLY=1 \
CONFIG=experiments/5/13_trajectory_distance_5-4_vs_5-11/config/final.tsv \
  experiments/5/13_trajectory_distance_5-4_vs_5-11/scripts/run_trajectory_distance.sh
```

## Run Pilot

GPU 0が空くまで待ってからpilotを開始する。

```bash
tmux new -s edm-exp5-13-pilot
experiments/5/13_trajectory_distance_5-4_vs_5-11/scripts/schedule_after_gpu_free.sh
```

pilot後、`results/pilot/trajectory_distances.svg`と`step_stats.tsv`を確認する。
RK4 4分割対8分割の距離が5-4／5-11の参照距離より十分小さいことを確認してから
finalへ進む。

## Run Final

```bash
tmux new -s edm-exp5-13-final
CONFIG=experiments/5/13_trajectory_distance_5-4_vs_5-11/config/final.tsv \
  experiments/5/13_trajectory_distance_5-4_vs_5-11/scripts/schedule_after_gpu_free.sh
```

## Monitor and Resume

```bash
tail -f experiments/5/13_trajectory_distance_5-4_vs_5-11/logs/trajectory-distance.log
tail -f experiments/5/13_trajectory_distance_5-4_vs_5-11/logs/pilot.log
tail -f experiments/5/13_trajectory_distance_5-4_vs_5-11/logs/final.log
```

同じコマンドを再実行すると、`per_seed_metrics.tsv`に存在するseedをスキップする。
`_SUCCESS`があるstageは実行ラッパー側でスキップする。

## Outputs

各stageの`results/<stage>/`へ次を保存する。

```text
run_config.json
per_seed_metrics.tsv
step_stats.tsv
trajectory_distances.svg
README.md
_SUCCESS
```

FIDは再計算せず、既存結果を解釈に使用する。

| experiment | FID-50k mean |
|---|---:|
| 5-4 variable AB | 3.57748 |
| 5-11 fixed AB | 1.99106 |
