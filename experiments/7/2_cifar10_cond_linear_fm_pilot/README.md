# Experiment 7-2: CIFAR-10 Conditional Linear FM Pilot

## 目的

Experiment 7-1で動作確認したlinear-path Flow Matching baselineを、修論で傾向を
判断できる20,000 kimgまで段階的に学習する。1,000 / 3,000 / 10,000 /
20,000 kimgで学習状態、固定seedの画像、FID-5kを保存し、EDMとの本比較へ進むかを
判断する。

本実験は7-1の50 kimg checkpointを既定の開始点とし、network重みとoptimizer状態を
引き継ぐ。EDMの`training-state-*.pt`は乱数状態とdata iterator位置を保存しないため、
未中断のrunとbit単位で同じ軌道になることは保証しない。

## 固定する定義

```text
z ~ N(0, I),  y ~ p_data,  t ~ Uniform[0, 1)
x_t = (1 - t) z + t y
v_target = y - z
dx/dt = v_theta(x, t),  x(0) = z
```

- independent couplingのlinear pathを使う。
- networkはvelocityを直接予測する。
- サンプリングは`t=0`（noise）から`t=1`（data）へ進む。
- pilotの既定samplerはEuler、18 steps = 18 NFEとする。
- `S_churn`、`rho`、sigma scheduleはFMには適用しない。

## 実験条件

| 項目 | 設定 |
| :--- | :--- |
| Dataset | CIFAR-10 32×32、class-conditional |
| Network | DDPM++、55,735,299 parameters |
| Training | batch 512、batch-gpu 128、Adam、LR 0.001 |
| EMA / regularization | EMA 0.5 Mimg、dropout 0.13、augment 0.12 |
| Precision / seed | FP32、seed 0、1 GPU |
| Stages | 1k / 3k / 10k / 20k kimg |
| Stage samples | seeds 0-63、FM Euler、18 steps |
| Stage metric | FID-5k、seeds 0-4999 |
| FID reference | `fid-refs/cifar10-32x32.npz` |

stage定義は[config/stages.tsv](config/stages.tsv)、実行本体は
[scripts/run_fm_linear_pilot.sh](scripts/run_fm_linear_pilot.sh)に置く。

## 実行方法

まずGPUを使わずに計画と入力ファイルを確認する。

```bash
PLAN_ONLY=1 \
  experiments/7/2_cifar10_cond_linear_fm_pilot/scripts/schedule_after_gpu_free.sh
```

本実験は長時間になるため、tmux内でGPU待機wrapperから起動する。

```bash
tmux new -s edm-exp7-2
GPU_INDEX=0 \
  experiments/7/2_cifar10_cond_linear_fm_pilot/scripts/schedule_after_gpu_free.sh
```

既定GPU名が異なる環境では`EXPECTED_GPU_NAME`を設定する。完全な新規初期化から
始める場合だけ`BASE_STATE=''`を指定する。

```bash
BASE_STATE='' GPU_INDEX=0 EXPECTED_GPU_NAME='' \
  experiments/7/2_cifar10_cond_linear_fm_pilot/scripts/schedule_after_gpu_free.sh
```

再実行時は7-2 run directory内の最新`training-state-*.pt`から再開し、完了済みの
health、sample、FIDをスキップする。現在のstage完了後に停止するには次を作る。

```bash
touch experiments/7/2_cifar10_cond_linear_fm_pilot/STOP_AFTER_CURRENT_STAGE
```

## 出力

```text
experiments/7/2_cifar10_cond_linear_fm_pilot/
  training-runs/cifar10-cond-ddpmpp-fm-linear-pilot/  checkpoints, stats
  images/samples-<kimg>-<sampler>-steps<N>-seeds-0-63/   fixed-seed samples
  images/fid-<kimg>-<sampler>-steps<N>-n5000-seeds-0-4999/ FID images
  logs/fm-linear-pilot.log                           master log
  results/fm_linear_pilot_health.tsv                 loss, speed, GPU memory
  results/fm_linear_pilot_fid_summary.tsv            FID and reproduction keys
```

checkpoint、生成画像、logは`.gitignore`対象で、リポジトリへcommitしない。

## EDMとの比較規約

FMとEDMでdataset、class conditioning、DDPM++容量、batch、optimizer、LR、EMA、
dropout、augmentation、FP32、seed、学習kimgを揃える。評価画像数とseed範囲も揃え、
まず18 NFEで比較する。

ただし、同じ`steps`は必ずしも同じNFEではない。既定EDM Heun 18 stepsは35 NFE、
FM Euler 18 stepsは18 NFEである。そのため論文用のsolver比較では少なくとも次の
2表を分ける。

1. 同じ既定設定: EDM Heun 18 steps（35 NFE）対FM Euler 18 steps（18 NFE）。
2. 同じ計算量: NFEを揃えたFID・生成時間比較。

FID-5kはpilot内の相対比較用であり、FID-50kの公開値と直接比較しない。20k時点で
lossが有限、画像が改善傾向、FIDが低下傾向ならfull実験を検討する。崩壊、改善停止、
またはEDMとの差が大きい場合は、長時間化する前にtime sampling、path、loss scalingを
再検討する。

7-1 smokeの実測6.54 sec/kimgから外挿すると、50→20,000 kimgの学習は約36.2時間。
これに4回のsample生成とFID-5kを加えるため、pilot全体はおよそ37〜40時間を見込む。

## 実行前チェック

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python -m compileall \
  train.py generate.py training/loss.py training/networks.py

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/test_fm_linear_baseline.py

/home/kyoukinkin/data/conda-envs/edm/bin/python train.py \
  --outdir=/tmp/edm-fm-pilot-dry-run \
  --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --precond=fm \
  --batch=512 --batch-gpu=128 --lr=.001 --ema=.5 --dropout=.13 \
  --augment=.12 --fp16=0 --seed=0 --duration=1 --dry-run
```
