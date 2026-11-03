# Sampler Experiment Scheduling Template

新しい sampler の FID 実験を追加するときは、GPUを直接取り合わないように
共通の待機ランチャを使う。

## Per-Experiment Wrapper

各実験ディレクトリに `scripts/schedule_after_gpu_free.sh` を置く。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/5/<experiment_dir>}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_<experiment>.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 5-X}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
```

## README Snippet

Run セクションには、通常実行に加えて待機実行も書く。

```bash
tmux new -s edm-exp5-X-<sampler>-scheduled
experiments/5/<experiment_dir>/scripts/schedule_after_gpu_free.sh
```

小さな確認実験を待機させる場合:

```bash
tmux new -s edm-exp5-X-<sampler>-smoke-scheduled
NUM_IMAGES=1024 \
CONFIG=experiments/5/<experiment_dir>/config/<smoke_config>.tsv \
  experiments/5/<experiment_dir>/scripts/schedule_after_gpu_free.sh
```

待機ログ:

```bash
tail -f experiments/5/<experiment_dir>/logs/schedule-after-gpu-free.log
```

## GPU設定

対象GPUは、PCごとの同期対象外ファイル `~/.config/edm/gpu.env` の `GPU_INDEX` と
`EXPECTED_GPU_NAME` で決まる（`scripts/load_gpu_config.sh` が読み込む）。設定がなければ
GPU処理を開始しない。共有コードやREADMEにGPU番号・GPU名を固定せず、コマンドラインの
環境変数でも上書きしない。別の設定ファイルを使う場合は `EDM_GPU_CONFIG` で指定する。

確認間隔だけは環境変数で変更できる（既定300秒）。

```bash
CHECK_INTERVAL_SEC=60 \
  experiments/5/<experiment_dir>/scripts/schedule_after_gpu_free.sh
```

## README冒頭

H1の直後に、目的を1〜2文で書く。結果が確定したら結論（主要なFIDなど）も同じ段落に加え、
グループREADMEを更新して `python scripts/build_experiment_index.py` でINDEXを再生成する。
