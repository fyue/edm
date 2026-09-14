# Experiment 8-3: Restart K=5 FID-50k over 3 Seed Ranges

## 目的

Experiment 8-2のFID-5kで確認した「Restartは学習不足checkpointでは改善し、学習後期
では悪化する」という傾向を、FID-50k × 3独立seed範囲で正式評価する。

parameter選択による交絡を避けるため、8-2と同一のRestart K=5設定、checkpoint、
ODE対照を変更せず使用する。

## 実験条件

- dataset/model: CIFAR-10 class-conditional EDM
- samples: 50,000/条件/seed範囲
- seed範囲: `0-49999`、`50000-99999`、`100000-149999`
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

checkpointは`5.5k`、`20k`、`120k`、`200k`。合計は12条件 × 3 seed範囲
= 36 FID評価、180万画像。8-2の実測から約3日、保存容量は約7–10 GiBを見込む。

## 主解析

各seed範囲内でpaired改善幅を計算する。

```text
improvement_vs_same_nfe = FID(equal-NFE ODE) - FID(Restart)
```

その3値について平均と標本標準偏差を報告する。正値はRestartによる改善を示す。
同じ18-step main baselineとの差は副解析とする。

## 計画確認

GPUを使わず、36個のcondition × seed範囲、checkpoint path、NFEを検証する。

```bash
PLAN_ONLY=1 \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

実checkpointで2枚だけ生成するsmoke test:

```bash
SAMPLES_ONLY=1 SAMPLE_SEEDS=0-1 SEED_FILTER="0-49999" \
CONDITION_FILTER="5p5k-restart-k5" GPU_INDEX=0 \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## 実行

tmux sessionを作成する。

```bash
tmux new -s edm-exp8-3
```

tmux内でGPU待機wrapperを実行する。

```bash
cd /data01/kyoukinkin/projects/edm
GPU_INDEX=0 \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

detachは`Ctrl-b`の後に`d`、再接続は次の通り。

```bash
tmux attach -t edm-exp8-3
```

## 部分実行

最初のseed範囲だけ実行する。

```bash
SEED_FILTER="0-49999" GPU_INDEX=0 \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

学習不足checkpointだけ先行する。

```bash
CHECKPOINT_FILTER="5.5k 20k" GPU_INDEX=0 \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

条件を限定する場合は、Restartだけでなく対応する2つのbaselineも含める。

```bash
CONDITION_FILTER="5p5k-ode-main18 5p5k-restart-k5 5p5k-ode-nfe55" GPU_INDEX=0 \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/schedule_after_gpu_free.sh
```

## 監視・停止・再開

```bash
tail -f experiments/8/3_restart_k5_fid50k_3seeds/logs/restart-k5-fid50k-3seeds.log
```

現在の1条件が終わったところで停止する。

```bash
touch experiments/8/3_restart_k5_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

再開前に停止fileを削除し、同じ実行コマンドを使う。結果TSVに記録済みの
condition・画像数・seed範囲・FID referenceはskipされる。画像が50,000枚揃って
いれば生成をskipしてFID計算から再開する。

## 集計とグラフ

runner終了時に自動生成する。手動で再生成する場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/1_restart_sampling_checkpoint_sweep/scripts/summarize_results.py \
  --input experiments/8/3_restart_k5_fid50k_3seeds/results/restart_fid_summary.tsv \
  --output experiments/8/3_restart_k5_fid50k_3seeds/results/restart_improvements.tsv

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/summarize_3seeds.py

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/3_restart_k5_fid50k_3seeds/scripts/plot_results.py
```

## 出力

```text
experiments/8/3_restart_k5_fid50k_3seeds/
  images/<condition>/fid-n50000-seeds-<range>/
  logs/
  results/restart_fid_summary.tsv
  results/restart_improvements.tsv
  results/restart_k5_fid50k_3seed_stats.tsv
  results/restart_k5_fid50k_3seed_overview.{png,svg}
```

生成画像とlogはcommitしない。
