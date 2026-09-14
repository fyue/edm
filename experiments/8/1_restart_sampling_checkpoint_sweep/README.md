# Experiment 8-1: Restart Sampling Checkpoint Sweep

## 目的

CIFAR-10 class-conditional EDMの学習途中checkpointに対し、Restart Samplingが通常の
ODE（EDM schedule + Heun）より生成品質を改善するか調べる。特に、学習不足なmodel
ほどRestartの改善幅が大きいという仮説を検証する。

この実験は保存済みcheckpointの評価だけを行い、再学習はしない。

## 実装

`generate.py --sampler=restart`は論文Algorithm 2と著者公開実装に合わせ、次を行う。

1. main EDM scheduleをHeunで下る。
2. 指定した`t_min`に最も近いmain schedule点へ到達したら、分散
   `t_max^2 - t_min^2`のGaussian noiseを加える。
3. `t_max`から丸め後の`t_min`まで、新しいEDM rho scheduleをHeunで下る。
4. 2–3を`K`回繰り返してmain processへ戻る。

CLI形式は著者公開実装と同じである。

```bash
--sampler=restart \
--restart_info='{"0":[3,2,0.06,0.30]}' \
--restart_noise=1
```

値は順に`[N_restart, K, t_min, t_max]`。複数levelもJSON objectへ追加できる。
空の`--restart_info='{}'`は通常のEDM Heunとbit単位で一致することをCPU testで確認する。

著者コード: https://github.com/Newbeeer/diffusion_restart_sampling

## Pilot設計

[config/pilot.tsv](config/pilot.tsv)では学習進行度の異なる4 checkpointを使う。

| checkpoint | 位置づけ |
|---:|---|
| 5.5k | 強い学習不足 |
| 20k | 学習初期 |
| 120k | 学習後期 |
| 200k | 自前runの最終checkpoint |

Restartはmain 18 stepsを固定し、公開CIFAR-10例の1-level区間
`N_restart=3, t_min=0.06, t_max=0.30`で`K=1,2,5`を比較する。

比較には二つの軸を用意する。

| 条件 | main steps | K | NFE | 比較目的 |
|---|---:|---:|---:|---|
| 通常Heun | 18 | 0 | 35 | 同じmain scheduleの基準 |
| Restart | 18 | 1 | 39 | 弱いRestart |
| 通常Heun | 20 | 0 | 39 | K=1との同NFE比較 |
| Restart | 18 | 2 | 43 | 論文の強いEDMに近い設定 |
| 通常Heun | 22 | 0 | 43 | K=2との同NFE比較 |
| Restart | 18 | 5 | 55 | 学習不足model向けの強い設定 |
| 通常Heun | 28 | 0 | 55 | K=5との同NFE比較 |

Heunの総NFEは次で検証され、条件表の値と異なる場合runnerは停止する。

```text
NFE = 2 * N_main - 1 + K * 2 * (N_restart - 1)
```

## 検証と計画表示

CPU unit testを実行する。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/test_restart_sampler.py
```

GPUを使用せず、全28条件、checkpoint、NFEを検証する。

```bash
PLAN_ONLY=1 \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/schedule_after_gpu_free.sh
```

実checkpointでFIDを計算せず2枚だけ生成するsmoke test:

```bash
SAMPLES_ONLY=1 SAMPLE_SEEDS=0-1 \
CONDITION_FILTER="5p5k-restart-k2" GPU_INDEX=0 \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/schedule_after_gpu_free.sh
```

## 実行

既定ではseeds `0-1023`のFID-1k pilotと、目視用のseeds `0-63`を生成する。
GPU 0が空くまで共通wrapperで待ってから開始する。

```bash
GPU_INDEX=0 \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/schedule_after_gpu_free.sh
```

まず学習初期だけを試す場合:

```bash
CHECKPOINT_FILTER="5.5k 20k" GPU_INDEX=0 \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/schedule_after_gpu_free.sh
```

特定条件だけを実行する場合:

```bash
CONDITION_FILTER="5p5k-ode-main18 5p5k-restart-k2 5p5k-ode-nfe43" GPU_INDEX=0 \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/schedule_after_gpu_free.sh
```

FID-5kへ進めるときは同じ条件表とrunnerを使う。

```bash
NUM_IMAGES=5000 FID_SEEDS=0-4999 SAMPLE_SEEDS=- GPU_INDEX=0 \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/schedule_after_gpu_free.sh
```

結果TSVに記録済みの同一条件・サンプル数・seed範囲・FID参照統計は再実行時に
skipする。現在の条件終了後に安全停止するには次を作成する。

```bash
touch experiments/8/1_restart_sampling_checkpoint_sweep/STOP_AFTER_CURRENT_CONDITION
```

## 集計

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/summarize_results.py
```

`results/restart_improvements.tsv`へ、同じ18-step main baselineからの改善と、同じ総NFEの
通常Heunからの改善を別々に出力する。正の値がRestartによるFID改善を表す。

グラフを生成する。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/plot_results.py
```

`results/restart_fid_curves.{png,svg}`はcheckpoint別のFID対NFE、
`results/restart_improvements.{png,svg}`は学習進行に対するRestartの改善幅を示す。

## 出力

```text
experiments/8/1_restart_sampling_checkpoint_sweep/
  images/<condition>/samples-seeds-0-63/
  images/<condition>/fid-n1024-seeds-0-1023/
  logs/
  results/restart_fid_summary.tsv
  results/restart_improvements.tsv
  results/restart_fid_curves.{png,svg}
  results/restart_improvements.{png,svg}
```

FID-1kはscreening用途に限る。傾向が残ったcheckpointとKだけをFID-5kへ進め、最終的な
主張には複数seed範囲のFID-50kを使う。
