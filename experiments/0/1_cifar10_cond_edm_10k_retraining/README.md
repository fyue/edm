# Experiment 0-1: CIFAR-10 Conditional EDM Exact 10k Retraining

全データ・学習seed 0で0→200kを1プロセスで連続学習し、10k刻み20点の均一なcheckpoint系列を作る。完了済みで、8-4・8-6・8-10〜8-13の評価対象（本実験ではFIDを評価しない）。

## 目的

修論本番用にCIFAR-10 class-conditional EDM/DDPM++をseed 0から200kまで再学習し、
10k刻みの均一なcheckpoint系列を作る。既存実験で評価した
`5.5k, 20k, 120k, 200k` の不均一な選択とは分離し、
`10k, 20k, ..., 200k` の20点を同じ学習系列から保存する。

この実験の範囲は学習とcheckpoint検証までとし、画像生成とFID評価は含めない。
20点の評価条件は、この系列の完成後に別実験として固定する。

## 厳密性の方針

- 0から200kまで、原則として1プロセスで連続学習する。
- 10kごとにプロセスを終了・再開しない。checkpoint保存のための再開による
  dataset iteratorと乱数列のリセットを避ける。
- `--checkpoint-kimgs` で指定点への到達をoptimizer updateごとに検出し、
  `network-snapshot-010000.pkl`から`network-snapshot-200000.pkl`までと、対応する
  `training-state-*.pt`を保存する。
- 通常のtick依存保存は`--snap=0 --dump=0`で無効化するため、約2.5k刻みの
  余分なcheckpointは作らない。

`batch=512`のため、10,000,000枚が512で割り切れない点では、指定枚数以上となる
最初のoptimizer update直後に保存する。これは最大384画像（0.384 kimg）の超過で、
ファイル名と評価上のcheckpointラベルは厳密に10k刻みとなる。部分batchを使って
optimizer条件を変えることはしない。

## 学習条件

| 項目 | 値 |
|---|---|
| Dataset | `datasets/cifar10-32x32.zip` |
| Conditional | class-conditional (`--cond=1`) |
| Architecture | DDPM++ (`--arch=ddpmpp`) |
| Preconditioning / loss | EDM (`--precond=edm`) |
| Duration | 200 Mimg = 200,000 kimg |
| Batch | 512 (`--batch-gpu=128`, 1 GPU) |
| Optimizer / LR | Adam / `0.001` |
| EMA | 0.5 Mimg |
| Dropout / Augment | 0.13 / 0.12 |
| Precision | FP32 |
| Seed | 0 |
| Checkpoints | 10k刻み、10k～200kの20点 |

既存の本番EDM学習と同じ条件を使用する。checkpoint一覧は
`config/checkpoint_kimgs.txt`を正とする。

## ファイル構成

```text
experiments/0/1_cifar10_cond_edm_10k_retraining/
  README.md
  config/checkpoint_kimgs.txt
  scripts/run_training.sh
  scripts/schedule_after_gpu_free.sh
  results/README.md
  results/checkpoint_manifest.tsv       # 完了後に生成
  logs/train.log                        # Git対象外
  training-runs/cifar10-cond-ddpmpp-edm-seed0-10k/  # Git対象外
```

## 実行前確認

GPUを使わず、設定と実行コマンドを確認する。

```bash
PLAN_ONLY=1 \
  experiments/0/1_cifar10_cond_edm_10k_retraining/scripts/run_training.sh
```

## 実行

共通のGPU待機スクリプトを経由してphysical GPU 0で起動する。

```bash
GPU_INDEX=0 \
  experiments/0/1_cifar10_cond_edm_10k_retraining/scripts/schedule_after_gpu_free.sh
```

進捗確認:

```bash
tail -f experiments/0/1_cifar10_cond_edm_10k_retraining/logs/train.log
```

## 中断と再開

10k以降でプロセスが異常終了した場合、同じコマンドを再実行すると、最後に完成した
`training-state-*.pt`と同名の`network-snapshot-*.pkl`から再開する。
最大で直近10k分を再実行する可能性がある。最初の10kより前に終了した場合は
再開可能なstateがないため、既存の不完全な`RUN_DIR`を別名で保全してから、新しい
`RUN_DIR`を指定して0からやり直す。

再開時は学習プロセスの乱数列・dataset iteratorが初期化されるため、完全な
bitwise連続性は失われる。修論本番系列では、可能な限り200kまで中断せずに
完走させる。

## 完了条件

10k～200kの全20点について、snapshotとstateの両方が非空であることをrunnerが
検証し、`results/checkpoint_manifest.tsv`を生成した時点で完了とする。

想定所要時間は約15日、20組のsnapshot/stateの保存容量は約16.6 GiB。
