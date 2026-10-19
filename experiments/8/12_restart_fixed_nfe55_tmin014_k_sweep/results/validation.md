# 実験8-12の実装検証（2026-10-12）

Pythonは`/home/kyoukinkin/data/conda-envs/edm/bin/python`を使用。

- `python -m compileall -q experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts`：成功。
- `python experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/test_experiment.py`：18件成功。CPUサンプラーの実NFEと実開始sigma、段階評価、停止・再開、SIGTERM/SIGKILL、一時PNG回収、FID再試行、原子的JSON保存、統合集計、無書き込みplanを確認。
- `bash -n`で`run.sh`と`schedule_after_gpu_free.sh`を検証：成功。
- `run.sh --help`：成功。
- 実設定・モデル・参照統計で`run.sh --plan-only --fid-stage all`：90条件、270評価、4,500,000枚。
- 同じplanに`--checkpoint-filter 20k-120k`：54条件、162評価、2,700,000枚。
- 同じplanに`--checkpoint-filter 140k-200k`：36条件、108評価、1,800,000枚。

GPU待機の呼び出しはCPUテストではmockで確認。中断処理は実際のダミー子プロセスで検証した。GPUスモーク、実GPU上の停止・再開、本FID評価は未実行。実行前にREADMEのスモーク手順を行う。
