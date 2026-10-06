# Experiment 8-6: Progressive Exact-10k Restart K=5 FID Evaluation

## 目的

Experiment 0-1で同一学習系列から保存した`10k, 20k, ..., 120k`の12 checkpointを、
Experiment 8-3と同じsampling条件で評価する。FID画像数は一度に50kまで生成せず、
`5k → 25k → 50k`の順に段階的に増やす。各段階の結果を確認してから次へ進み、不要なら
その時点で実験を終了できるようにする。

## 実験条件

- dataset/model: CIFAR-10 class-conditional EDM/DDPM++（Experiment 0-1）
- checkpoints: `10k, 20k, ..., 120k`（12点）
- FID stages: `5,000`、`25,000`、`50,000`
- nested seed範囲: `0-4999`、`0-24999`、`0-49999`
- Restart: main 18 steps、`N_restart=3`、`K=5`、`t_min=0.06`、`t_max=0.30`
- Restart noise multiplier: `1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- GPU: 1

各checkpointで次の3条件を評価する。

| 条件 | main steps | K | NFE | 目的 |
|---|---:|---:|---:|---|
| ODE main baseline | 18 | 0 | 35 | 同じmain scheduleとの比較 |
| Restart | 18 | 5 | 55 | 評価対象 |
| ODE equal-NFE baseline | 28 | 0 | 55 | 同じ計算量との比較 |

各段階は12 checkpoint × 3条件 = 36 FID評価。画像はconditionごとの共通poolへ保存し、
次段階では不足分だけを追加する。

| FID stage | poolへ追加するseed | 追加画像数/condition |
|---:|---|---:|
| 5k | `0-4999` | 5,000 |
| 25k | `5000-24999` | 20,000 |
| 50k | `25000-49999` | 25,000 |

## 所要時間と保存容量の見込み

Experiment 8-2のFID-5kおよびExperiment 8-3のFID-50k実測に基づく。

| 到達段階 | 今回の追加時間 | 累計時間 |
|---:|---:|---:|
| 5k | 約7時間 | 約7時間 |
| 25k | 約26時間 | 約33時間（約1日9時間） |
| 50k | 約33時間 | 約66時間（約2日18時間） |

50kまで進めた場合、各段階でFIDを再計算するため、直接50kを評価する約61時間に対して
約4〜5時間の追加負担となる。生成画像は再利用するため、最終保存容量は直接50kと同じ
約7〜10 GiB。待機時間と段階間の判断時間は上記に含まない。

## 主解析

各FID stage・checkpointについて次のpaired改善幅を報告する。

```text
improvement_vs_same_nfe = FID(equal-NFE ODE) - FID(Restart)
```

正値はRestartによる改善を示す。同じ18-step main baselineとの差は副解析とする。
FID-5k、FID-25k、FID-50kは標本数によるバイアスが異なるため、段階をまたいだFID絶対値
の直接比較ではなく、各段階内のpaired差を進行判断に使う。

## 計画確認

GPUを使わず、各段階の36条件、追加seed、checkpoint path、NFEを検証する。

```bash
FID_STAGE=5000 PLAN_ONLY=1 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh

FID_STAGE=25000 PLAN_ONLY=1 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh

FID_STAGE=50000 PLAN_ONLY=1 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

実checkpointで2枚だけ生成するsmoke test:

```bash
FID_STAGE=5000 SAMPLES_ONLY=1 SAMPLE_SEEDS=0-1 \
CONDITION_FILTER="10k-restart-k5" GPU_INDEX=0 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

## 段階実行

### Stage 1: FID-5k

```bash
tmux new -s edm-exp8-6
cd /data01/kyoukinkin/projects/edm
FID_STAGE=5000 GPU_INDEX=0 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

結果を確認し、続行する場合だけ次へ進む。

### Stage 2: FID-25k

5k画像を再利用し、追加20kだけを生成する。選択した全conditionに対応するFID-5k記録が
存在しない場合は開始しない。

```bash
FID_STAGE=25000 GPU_INDEX=0 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

### Stage 3: FID-50k

25k画像を再利用し、追加25kだけを生成する。対応するFID-25k記録を必須とする。

```bash
FID_STAGE=50000 GPU_INDEX=0 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

tmuxのdetachは`Ctrl-b`の後に`d`、再接続は次の通り。

```bash
tmux attach -t edm-exp8-6
```

## 部分実行

checkpointを限定する。次段階でも同じfilterを指定する。

```bash
FID_STAGE=5000 CHECKPOINT_FILTER="10k 20k 30k" GPU_INDEX=0 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

条件を限定する場合は、Restartと対応する2つのbaselineを同時に指定する。

```bash
FID_STAGE=5000 \
CONDITION_FILTER="10k-ode-main18 10k-restart-k5 10k-ode-nfe55" GPU_INDEX=0 \
  experiments/8/6_exact_10k_restart_k5_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

## 監視・停止・再開

```bash
tail -f experiments/8/6_exact_10k_restart_k5_fid50k_1seed/logs/progressive-fid.log
```

現在の1条件が終わったところで停止する。

```bash
touch experiments/8/6_exact_10k_restart_k5_fid50k_1seed/STOP_AFTER_CURRENT_CONDITION
```

再開前に停止ファイルを削除し、同じstageの実行コマンドを使う。結果TSVに記録済みの
condition・画像数・seed範囲・FID referenceはskipされる。中断したstageのseed範囲は
決定的に再生成される場合があるが、それ以前のstageの画像は再生成しない。

## 出力

```text
experiments/8/6_exact_10k_restart_k5_fid50k_1seed/
  images/<condition>/pool-seeds-0-49999/
  logs/
  results/restart_fid_summary.tsv
  results/restart_improvements.tsv
  results/restart_k5_fid5000_overview.{png,svg}
  results/restart_k5_fid25000_overview.{png,svg}
  results/restart_k5_fid50000_overview.{png,svg}
```

生成画像とlogはcommitしない。
