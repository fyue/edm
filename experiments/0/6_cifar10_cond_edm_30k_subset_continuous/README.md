# Experiment 0-6: CIFAR-10 Conditional EDM on a 30k Subset, Continuous Training

0-2と同じ3万枚を、0-1と同じく途中で止めずに1プロセスで100kまで学習し、0-2のFID悪化が再開による乱数の再生に影響されていないかを切り分ける。このPCには10kのsnapshot/stateが保存されており、100kまでの完了は未確認。

## 背景

`train.py --resume`が復元するのは重みとoptimizerの状態だけで、乱数の位置は復元しない。
`--seed`を固定すると、`training/training_loop.py`の乱数と`InfiniteSampler`が再開のたびに
先頭へ戻る。

0-2は途中FID評価のために10kごとに学習を止めて再開しており、50k以降の区間どうしで
lossの揺れの相関が0.99になる。無停止の0-1は約0である（`scripts/check_stream_replay.py`）。
そのため0-1と0-2の差には、画像枚数の違いと再生の有無が混ざっている。FIDへの影響は未確定。

2026-11-05時点の計画表示は10kのstateからの再開を検出する。再開した場合は乱数seedを変更し、単一の連続プロセスでの学習ではなくなったことを警告する。連続学習の対照として使う場合は、実際の起動・再開履歴を確認する。

## 条件

0-1との違いは学習データ（3万枚）と学習量（100k）だけ。0-2との違いは、途中で止めないことと、checkpointが10k刻みであること。

| 項目 | 値 |
|---|---|
| Dataset | `datasets/cifar10-32x32.zip`から30,000枚（0-2と同じ画像） |
| Subset抽出 | `--dataset-max-size=30000 --dataset-random-seed=0` |
| Conditional / model | class-conditional / DDPM++ / EDM |
| Batch / GPU数 | 512 / 1 GPU（`--batch-gpu=128`） |
| Optimizer / LR | Adam / `0.001` |
| EMA | 0.5 Mimg |
| Dropout / Augment | 0.13 / 0.12 |
| Precision / training seed | FP32 / seed 0 |
| 学習 | 0→100,000 kimgを1プロセスで連続実行。途中のFID評価なし |
| Checkpoint | 10k刻みの10点、snapshot/stateの両方（約8.3 GiB）。`config/checkpoint_kimgs.txt` |
| FID評価（学習後） | 10k刻みの10点、EDM Heun 18 steps（NFE 35）、churn 0、seeds `0-49999`、50,000枚 |
| FID参照 | 全CIFAR-10の`fid-refs/cifar10-32x32.npz` |

実測6.5 sec/kimgから、学習は約7.5日、FID評価は全10点で約13時間の見込み。

## 実行前確認

GPU設定の読み込み・待機・学習は行わない。

```bash
PLAN_ONLY=1 experiments/0/6_cifar10_cond_edm_30k_subset_continuous/scripts/run_training.sh
PLAN_ONLY=1 experiments/0/6_cifar10_cond_edm_30k_subset_continuous/scripts/run_fid.sh
```

## 学習

```bash
tmux new -s edm-exp0-6
experiments/0/6_cifar10_cond_edm_30k_subset_continuous/scripts/schedule_after_gpu_free.sh
```

共通の`scripts/wait_for_gpu_and_run.sh`でGPUが空くまで待機する。対象GPUはローカル設定（`~/.config/edm/gpu.env`の`GPU_INDEX`・`EXPECTED_GPU_NAME`）で決まる。

進捗確認:

```bash
tail -f experiments/0/6_cifar10_cond_edm_30k_subset_continuous/logs/train.log
```

## 学習後のFID評価

```bash
tmux new -s edm-exp0-6-fid
experiments/0/6_cifar10_cond_edm_30k_subset_continuous/scripts/schedule_fid_after_gpu_free.sh
```

評価手順は0-2と同じで、結果も同じ形式で`results/fid_summary.tsv`に保存する。記録済みの点はスキップする。

## 再生していないことの確認

相関が0付近なら再生は起きていない。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/0/6_cifar10_cond_edm_30k_subset_continuous/scripts/check_stream_replay.py
```

## 異常終了した場合

10k以降で止まった場合、同じコマンドを再実行すると最新のstateから再開する。最大で直近10k分を学習し直す。
再開時は再生を避けるため、seedを`再開kimg`（30kからなら30000）に変える。ただし1プロセスの連続学習ではなくなるので、そのことを結果に明記する。

最初の10kより前に止まった場合は再開できないので、不完全な`training-runs/`を別名で保全してからやり直す。

## 結果の読み方

- 50k前後からFIDが悪化する：悪化は再開の再生では説明できず、データ不足の影響と考えられる。このcheckpointでK走査をやり直す
- 悪化しない、または大きく弱まる：0-2の悪化には再生の影響があり、8-14と修論のデータ不足の章を見直す

学習seedは系列ごとに1つなので、小さな差は学習の揺らぎと区別できない。

## 出力

- 学習: `training-runs/cifar10-cond-ddpmpp-edm-seed0-subset30k-continuous/`
- FID画像: `fid-evals/cifar10-cond-ddpmpp-edm-seed0-subset30k-continuous/`
- ログ: `logs/train.log`、`logs/fid.log`
- checkpoint一覧: `results/checkpoint_manifest.tsv`
- FID集計: `results/fid_summary.tsv`

すべて本実験ディレクトリからの相対パス。大容量の学習成果物・生成画像はコミットしない。
