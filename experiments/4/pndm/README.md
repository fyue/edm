# Experiment 4 PNDM: checkpoint-wise S_churn sensitivity

## 目的

Experiment 4 の EDM sampler に対する `S_churn` 感度実験を、EDM座標系へ移植した
PNDM sampler（`generate.py --sampler=pndm`）で再現する。

既存の `experiments/4/stage0`, `stage1`, `stage2` は移動・変更せず、このフォルダに
PNDM版の設定、スクリプト、結果を分離して保存する。

## 実験フロー

| stage | 目的 | samples | seeds | 実行条件 |
|---|---|---:|---|---|
| Stage 0 | 低コストな全体傾向の確認 | 1,024 | `0-1023` | 4 checkpoints × `S_churn=0-7` |
| Stage 1 | checkpoint全体のスクリーニング | 5,000 | `0-4999` | 8 checkpoints × `S_churn=0-7` |
| Stage 2-1 | EDM版と対応する代表条件の正式確認 | 50,000 | `0-49999` | Stage 1確認後に実行 |
| Stage 2-2 | 代表ペアの追加seed確認 | 50,000 | 追加2範囲 | Stage 2-1確認後に実行 |

Stage 2 の設定は EDM版と直接比較できる候補を初期値として置いている。PNDMの
Stage 1結果がEDMと異なる場合は、本実行前にStage 2のconfigを更新する。

## Stage 0・1で使用するcheckpoint

自前学習checkpointは、すべて次のtraining runから読み込む。

```text
experiments/1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/
  training-runs/cifar10-cond-ddpmpp-edm-managed/
```

各 `.pkl` に保存されたEMA networkを画像生成に使用する。

### Stage 0

Stage 0では、学習初期・中期・後期と公式pretrainedを代表する次の4モデルを使う。

| checkpoint label | 使用するnetwork |
|---|---|
| `10k` | `network-snapshot-010000.pkl` |
| `40k` | `network-snapshot-040000.pkl` |
| `120k` | `network-snapshot-120000.pkl` |
| `pretrained` | NVIDIA公式 `edm-cifar10-32x32-cond-vp.pkl` |

### Stage 1

Stage 1では学習進行に対する感度を細かく見るため、次の8モデルを使う。
`5.5k` と `15.0k` は表示上の丸めたラベルで、実際のsnapshot番号も併記する。

| checkpoint label | 使用するnetwork | 実際のsnapshot番号 |
|---|---|---:|
| `5.5k` | `network-snapshot-005509.pkl` | `5509` |
| `10k` | `network-snapshot-010000.pkl` | `10000` |
| `15.0k` | `network-snapshot-015018.pkl` | `15018` |
| `20k` | `network-snapshot-020000.pkl` | `20000` |
| `40k` | `network-snapshot-040000.pkl` | `40000` |
| `80k` | `network-snapshot-080000.pkl` | `80000` |
| `120k` | `network-snapshot-120000.pkl` | `120000` |
| `pretrained` | NVIDIA公式 `edm-cifar10-32x32-cond-vp.pkl` | — |

公式pretrained modelは次のURLから読み込む。

```text
https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

## 比較条件

- dataset/model: CIFAR-10 class-conditional
- sampler: `pndm`
- steps: `18`
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- GPU: デフォルトではphysical GPU 0

`steps=18` をEDM版と揃えるため、これは「同一steps」の比較である。PNDMは開始時に
RK4評価を使うため、EDM samplerとNFEは一致しない。等NFE比較はこの実験とは分離する。

## 実行順

各stageで、まず `PLAN_ONLY=1` により条件を確認する。

```bash
PLAN_ONLY=1 experiments/4/pndm/stage0/1_fid1k_churn_sweep/scripts/run.sh
PLAN_ONLY=1 experiments/4/pndm/stage1/1_fid5k_churn_sweep/scripts/run.sh
PLAN_ONLY=1 experiments/4/pndm/stage2/1_fid50k_selected_churn/scripts/run.sh
PLAN_ONLY=1 experiments/4/pndm/stage2/2_fid50k_extra_seeds/scripts/run.sh
```

Stage 0とStage 1の結果を確認してから、Stage 2のconfigを確定する。

## 4 stageの自動連続実行

4つのrunnerは次の順番で自動実行できる。

```text
Stage 0 → Stage 1 → Stage 2-1 → Stage 2-2 → 3 seed集計
```

全条件の計画だけを確認する。この操作ではGPU待機、画像生成、ログ作成を行わない。

```bash
PLAN_ONLY=1 experiments/4/pndm/scripts/run_all_stages.sh
```

Stage 2のconfigを確認した後、全stageを連続実行する。

```bash
CONFIRM_STAGE2_CONFIG=1 experiments/4/pndm/scripts/run_all_stages.sh
```

Stage 0とStage 1だけを連続実行し、その結果を見てStage 2条件を決める場合:

```bash
RUN_STAGE2_1=0 RUN_STAGE2_2=0 \
  experiments/4/pndm/scripts/run_all_stages.sh
```

その後、Stage 2だけを連続実行する。

```bash
RUN_STAGE0=0 RUN_STAGE1=0 CONFIRM_STAGE2_CONFIG=1 \
  experiments/4/pndm/scripts/run_all_stages.sh
```

途中で失敗しても、各stageのsummaryに記録済みの条件は再実行時にskipされる。

## GPU空き待機スケジュール

Experiment 5と同様に、physical GPU 0上のcompute processがなくなるまで待機してから
連続runnerを開始できる。tmux内での実行を推奨する。

```bash
tmux new -s edm-exp4-pndm-scheduled
CONFIRM_STAGE2_CONFIG=1 \
  experiments/4/pndm/scripts/wait_for_gpu_and_run.sh
```

Stage 0とStage 1だけをGPU空き後に実行する場合:

```bash
tmux new -s edm-exp4-pndm-screening
RUN_STAGE2_1=0 RUN_STAGE2_2=0 \
  experiments/4/pndm/scripts/wait_for_gpu_and_run.sh
```

待機ログと全体進行ログ:

```text
experiments/4/pndm/logs/schedule-after-gpu-free.log
experiments/4/pndm/logs/run-all-stages.log
```

既定値はExperiment 5と同じである。

- `GPU_INDEX=0`
- `EXPECTED_GPU_NAME="Quadro RTX 8000"`
- `CHECK_INTERVAL_SEC=300`

GPU名チェックを無効にする場合は `EXPECTED_GPU_NAME=""` を指定する。

## Experiment 5 の既存結果

`experiments/5/1_pndm_fid50k_3seeds` には、pretrained、`S_churn=0`、
18 steps、3 seed範囲のPNDM FID-50kが存在する。Stage 2ではこれを
`stage2/config/reused_experiment5_pretrained.tsv` から参照し、同一条件を再生成しない。

## 共通機能

```text
scripts/run_fid_churn_sweep.sh  生成、FID、再開、停止
scripts/plot_fid_churn.py       checkpoint別のFID曲線
scripts/summarize_paired_seeds.py
                                3 seed平均とpaired improvement
```

各条件はsummary TSVに記録され、同じ条件を再実行した場合はスキップされる。
現在の条件終了後に止める場合は、対象実験フォルダへ
`STOP_AFTER_CURRENT_CONDITION` を作成する。

## 保存方針

- `config/`, `scripts/`, `README.md`, summary TSV、最終集計はGit管理対象とする。
- 数十万～数百万枚になる `images/` と実行時の `logs/` はGit管理しない。
- 既存EDM実験およびExperiment 5の成果物は変更しない。
