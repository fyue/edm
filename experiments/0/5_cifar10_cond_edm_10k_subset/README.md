# Experiment 0-5: CIFAR-10 Conditional EDM on a 10k Subset

0-2と同じ条件で学習データを10,000枚に減らし、30k（kimg）まで学習してデータ量の影響を調べる。30kまで完了し、FID-50kは10k 4.63680、20k 2.82997（最良）、30k 3.77162だった。

0-2と同じ学習・評価条件を使い、学習サブセットを30,000枚から10,000枚へ変更する。
seed 0から新規学習し、30,000 kimg（30 Mimg）まで学習する。
ここでサブセットの10kは画像枚数、学習量の30kはkimg単位を表す。

結果の出典：[fid_summary.tsv](results/fid_summary.tsv)、[run_status.tsv](results/run_status.tsv)。生成seed範囲は全評価で`0-49999`。

## 条件

| 項目 | 値 |
|---|---|
| Dataset | `datasets/cifar10-32x32.zip`から10,000枚 |
| Subset抽出 | `--dataset-max-size=10000 --dataset-random-seed=0` |
| Conditional / model | class-conditional / DDPM++ / EDM |
| Batch / GPU数 | 512 / 1 GPU（`--batch-gpu=128`） |
| Optimizer / LR | Adam / `0.001` |
| EMA | 0.5 Mimg |
| Dropout / Augment | 0.13 / 0.12 |
| Precision / training seed | FP32 / seed 0 |
| 学習終了点 | 30,000 kimg |
| Checkpoint | 2,500 kimgごと、snapshot/stateの両方（全12組） |
| FID評価点 | 10,000 / 20,000 / 30,000 kimg |
| FID生成 | EDM Heun 18 steps（NFE 35）、churn 0、seeds `0-49999`、50,000枚 |
| FID参照 | 全CIFAR-10の`fid-refs/cifar10-32x32.npz` |
| 早期終了 | 無効（`EARLY_STOP=0`）。30,000 kimgまで学習する |

subsetは0-2と同じ決定論的ランダム抽出を用い、クラス別件数の均等化は行わない。
0-2のrunnerとFID画像検証・集計処理を読み込み、実験条件の既定値と出力先を
0-5用に設定する。0-2のcheckpointや結果は再利用しない。

## 実行前確認

GPU設定の読み込み・待機・学習を行わずに、全3区間のコマンドを表示する。

```bash
PLAN_ONLY=1 experiments/0/5_cifar10_cond_edm_10k_subset/scripts/schedule_after_gpu_free.sh
```

## 実行

```bash
experiments/0/5_cifar10_cond_edm_10k_subset/scripts/schedule_after_gpu_free.sh
```

共通の`scripts/wait_for_gpu_and_run.sh`でGPUが空くまで待機する。
`scripts/load_gpu_config.sh`がローカルの`~/.config/edm/gpu.env`（または
`EDM_GPU_CONFIG`）から`GPU_INDEX`と`EXPECTED_GPU_NAME`を読み込み、指定GPUを使う。
設定がない場合やGPU名が一致しない場合は起動しない。

0-2と同じく10,000 kimgごとに学習プロセスを終了し、FID評価後にstateから再開する。
再開時には乱数列・dataset iteratorが初期化されるため、連続した単一プロセスの
学習とはbitwise一致しない。batch 512による端数処理も0-2と同じで、指定kimg以上の
最初のupdateで保存する。

## 出力と再開

- 学習: `training-runs/cifar10-cond-ddpmpp-edm-seed0-subset10k/`
- FID画像: `fid-evals/cifar10-cond-ddpmpp-edm-seed0-subset10k/`
- ログ: `logs/train-and-fid.log`
- FID集計: `results/fid_summary.tsv`
- 実行状態: `results/run_status.tsv`

すべて本実験ディレクトリからの相対パス。同じ実行コマンドで最新の完成済み
snapshot/stateペアから再開し、記録済みFIDをスキップする。再開前にsubset件数・
subset seed・training seed・batchとcheckpointの整合性を検証する。
FID画像はseedと配置を検証し、不足分だけを生成する。

大容量の学習成果物・生成画像はコミットしない。
