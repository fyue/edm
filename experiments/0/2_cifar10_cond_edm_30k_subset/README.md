# Experiment 0-2: CIFAR-10 Conditional EDM on a 30k Subset

## 目的

CIFAR-10の50,000件から固定seedで30,000件を抽出し、class-conditional
EDM/DDPM++を学習する。2,500 kimgごとにcheckpointを保存し、10,000 kimgごとにFID50kを評価する。
改善が止まった時点で学習を早期終了する。最大量は既定で60,000 kimgとする。

## 条件

| 項目 | 値 |
|---|---|
| Dataset | `datasets/cifar10-32x32.zip`から30,000件 |
| Subset抽出 | `--dataset-max-size=30000 --dataset-random-seed=0` |
| Conditional | class-conditional (`--cond=1`) |
| Architecture | DDPM++ / EDM |
| Batch | 512 (`--batch-gpu=128`, 1 GPU) |
| Optimizer / LR | Adam / `0.001` |
| EMA | 0.5 Mimg |
| Dropout / Augment | 0.13 / 0.12 |
| Precision / training seed | FP32 / seed 0 |
| Checkpoint | 2,500 kimgごと（snapshot/stateの両方） |
| FID評価 | 10,000 kimgごと |
| FID | EDM 18 steps、churn 0、seeds `0-49999` |
| FID参照 | 全CIFAR-10の`fid-refs/cifar10-32x32.npz` |
| 最大量 | 60,000 kimg |

subsetはclassごとの件数を厳密に均等化せず、`training.dataset.Dataset`の既存機能で
決定論的にランダム抽出する。抽出件数とseedは各snapshot内の`dataset_kwargs`と
`training_options.json`にも保存される。

## 早期終了

最初のFIDをbestとし、それ以後はbestを`FID_MIN_DELTA`以上改善したときだけbestを
更新する。改善不足が`FID_PATIENCE`回続いたら、直近checkpointのFID評価完了後に
終了する。既定値は`FID_MIN_DELTA=0.1`、`FID_PATIENCE=2`なので、最短の停止点は
30,000 kimgである。

`EARLY_STOP=0`で早期終了を無効化できる。早期終了後に最大量まで続ける場合も、同じ
実行コマンドへ`EARLY_STOP=0`を付ければ最後のcheckpointから再開する。

## 実行前確認

GPUを使わず、10k～60kの学習・生成・FIDコマンドを表示する。

```bash
PLAN_ONLY=1 \
  experiments/0/2_cifar10_cond_edm_30k_subset/scripts/schedule_after_gpu_free.sh
```

最大量を変更する場合は10,000の倍数を指定する。

```bash
MAX_KIMG=80000 PLAN_ONLY=1 \
  experiments/0/2_cifar10_cond_edm_30k_subset/scripts/schedule_after_gpu_free.sh
```

## 実行

共通のGPU待機スクリプトを経由し、ローカルの`~/.config/edm/gpu.env`に
指定したGPUで起動する。設定の読み込みは`scripts/load_gpu_config.sh`が行う。

```bash
experiments/0/2_cifar10_cond_edm_30k_subset/scripts/schedule_after_gpu_free.sh
```

進捗と結果:

```bash
tail -f experiments/0/2_cifar10_cond_edm_30k_subset/logs/train-and-fid.log
cat experiments/0/2_cifar10_cond_edm_30k_subset/results/fid_summary.tsv
cat experiments/0/2_cifar10_cond_edm_30k_subset/results/run_status.tsv
```

## 中断・再開と再現性

runnerを再実行すると、完成済みのcheckpointとFIDをスキップする。FID画像のseedと
配置を検証し、50,000枚揃っていれば全体を再利用し、不足していれば不足seedだけを
生成する。想定外のseed、ファイル名、配置があれば誤ったFIDを防ぐため停止する。
checkpointのsnapshot/stateペアが欠けている場合や、既存runのsubset件数・subset seed・training
seed・batchが今回の条件と異なる場合は、混在を避けるため停止する。

中断後は最新の2,500 kimg刻みのsnapshot/stateペアから次の10,000 kimg評価点まで
再開する。例えば22,500 kimgのstateがあれば、25,000・27,500・30,000 kimgで保存し、
30,000 kimgでFIDを評価する。保存完了直後の中断なら再実行量はほぼ0、
保存間隔内での中断なら最大約2,500 kimgとなる。batch 512のため、指定点以上となる
最初のoptimizer updateで保存する（最大0.512 kimg未満の超過）。

改修前の10,000・20,000 kimgだけが保存されたrunも再開できる。
過去の2,500 kimg刻みのcheckpointを再学習で補完せず、再開後の保存点から適用する。
保存数は従来の4倍になり、snapshot/stateの1組は約0.83 GiBを使用する。

1 GPUで各FIDを学習の間に実行するため、学習プロセスは10,000 kimgごとに正常終了し、
FID評価後に`training-state`から再開する。区間内の2,500 kimgごとの保存では
プロセスを終了しない。重みとoptimizerは復元されるが、乱数列とdataset
iteratorは区間ごとに初期化されるため、0から60,000 kimgまで単一プロセスで走らせる
学習とはbitwise一致しない。

## ファイル構成

```text
experiments/0/2_cifar10_cond_edm_30k_subset/
  README.md
  scripts/evaluate_plateau.py
  scripts/inspect_image_pool.py
  scripts/run_train_and_fid.sh
  scripts/test_checkpoint_schedule.sh
  scripts/schedule_after_gpu_free.sh
  results/README.md
  results/fid_summary.tsv       # 実行時に生成
  results/run_status.tsv        # 実行時に生成
  logs/                         # Git対象外
  training-runs/                # Git対象外
  fid-evals/                    # Git対象外
```
