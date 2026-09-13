# Experiment 7-1: CIFAR-10 Class-Conditional Linear Flow Matching Baseline

## Purpose

CIFAR-10 class-conditionalを対象に、EDMと同じDDPM++構造・学習設定を使う
linear-path flow matching (FM) baselineを追加する。EDMとFMの違いを
「経路と学習目標」だけに絞り、architecture・batch・LR・EMA・augmentationは
既存のEDM実験 (`experiments/1`, `experiments/3`) と揃える。

## Formulation

独立に標本化したGaussian noise `z ~ N(0,I)` と画像 `y` を線形経路で結ぶ。

```text
x_t = (1 - t) z + t y,    t ~ U[0,1)
v_target = d x_t / dt = y - z
```

- `t = 0` がnoise、`t = 1` がデータ。**時間の向きはEDMと逆**で、samplingは
  `t = 0 -> 1` へ積分する。
- networkは速度場 `v(x_t, t)` を直接出力する。EDMのような
  preconditioning (`c_skip`, `c_out`, `c_in`) もloss weightingも使わず、
  重み1のvelocity MSEをそのまま最小化する。
- 時刻条件はDDPM++のtimestep embeddingへ `c_noise = 999 * t` として渡す
  (`M = 1000`)。
- samplingは `x(0) = z` から始める。EDM samplerのような `sigma_max` による
  latentのスケーリングは行わない。

Lossの初期値は「velocityを0と予測した場合」の理論値

```text
E||y - z||^2 / dim = sigma_data^2 + 1 = 0.5^2 + 1 = 1.25
```

に一致する。学習初期のlossが1.25付近から下がることが健全性の目安になる。

## Implementation

| File | Change |
| :--- | :--- |
| [training/loss.py](../../../training/loss.py) | `FMLoss` を追加。`sample_noise` / `sample_path` / `velocity_target` に分けてあり、Minibatch OTを入れる場合は `sample_noise` のcouplingを差し替える。 |
| [training/networks.py](../../../training/networks.py) | `FMPrecond` を追加。既存SongUNetをそのまま使い、velocityを直接出力。`prediction_type='velocity'`、`path='linear'`、`t_min=0`、`t_max=1`、`M=1000` をcheckpointへ保存。 |
| [train.py](../../../train.py) | `--precond=fm` を追加 (`FMPrecond` + `FMLoss(path='linear')`)。 |
| [generate.py](../../../generate.py) | `--sampler=fm-euler` / `fm-heun` と `check_sampler_compatibility()` を追加。 |
| [docs/train-help.txt](../../../docs/train-help.txt), [docs/generate-help.txt](../../../docs/generate-help.txt) | `--help` の保存資料を再生成。 |

`FMPrecond` は `sigma_min` / `sigma_max` を持たないため、EDM系samplerから見ると
明確に別種のcheckpointになる。checkpointとsamplerの取り違えは生成前に
`click.ClickException` で弾く。

### Sampler と NFE

`--steps=N` は `t=0` から `t=1` までの**区間数**を指す。

| Sampler | 更新式 | NFE |
| :--- | :--- | :--- |
| `fm-euler` | `x_{n+1} = x_n + h v(x_n, t_n)` | `N` |
| `fm-heun` | 各区間でHeun補正、最後の区間のみEuler (EDM samplerと同じ) | `2N - 1` |

`--steps=18` なら Euler 18 NFE、Heun 35 NFE で、EDM samplerの既定 (18 steps /
35 NFE) とNFEを揃えて比較できる。

### 拒否する組合せ

- FM samplerにEDM denoiser checkpointを渡す / EDM系samplerにvelocity
  checkpointを渡す。
- FM samplerと ablation option (`--solver` / `--disc` / `--schedule` / `--scaling`)。
- FM samplerと非既定の `--sigma_min` / `--sigma_max` / `--rho` / `--S_churn` /
  `--S_min` / `--S_max` / `--S_noise`。

## Files

```text
experiments/7/1_cifar10_cond_linear_fm_baseline/
  README.md
  config/
    profiles.tsv                     # smoke / pilot / full の段階定義
  scripts/
    run_fm_linear_baseline.sh        # 再開可能な段階学習runner
    schedule_after_gpu_free.sh       # scripts/wait_for_gpu_and_run.sh の薄いラッパー
    test_fm_linear_baseline.py       # FM実装のunit test (GPU・データセット不要)
    verify_fm_checkpoint.py          # checkpoint再読込とloss/gradient健全性の検証
  results/
    fm_linear_baseline_health.tsv    # stageごとのloss・速度・GPUメモリ
    fm_linear_baseline_fid_summary.tsv # stageごとのFID (pilot/full用、現在はheaderのみ)
  training-runs/
    cifar10-cond-ddpmpp-fm-linear/   # 学習成果物 (checkpoint, log, stats)
```

学習成果物は同ディレクトリ内の
`experiments/7/1_cifar10_cond_linear_fm_baseline/training-runs/cifar10-cond-ddpmpp-fm-linear/`
に置き、実験ごとの成果物を1か所に集約している (トップレベルの `training-runs/`
には置かない)。生成画像は同 `images/`、ログは同 `logs/` に出る。いずれも
`.gitignore` の `training-*` / `experiments/**/images/` / `experiments/**/logs/`
規則でコミット対象外。

## Profiles

| Profile | Stages (kimg) | tick/snap/dump | Stage FID | Final FID | 検証 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `smoke` | 50 | 10 / 1 / 1 | なし | なし | 決定性・checkpoint再読込 |
| `pilot` | 1000, 3000, 10000, 20000 | 50 / 50 / 50 | FID-5k (seeds 0-4999) | なし | なし |
| `full` | 40000, 80000, 120000, 160000, 200000 | 50 / 50 / 50 | FID-5k (seeds 0-4999) | FID-50k × 3 seed範囲 | なし |

各stageで sample 64枚 (seeds 0-63, `fm-euler`, 18 steps) を生成し、
health (loss・sec/kimg・GPUメモリ) を記録する。**smokeが通ってもpilot/fullは
自動起動しない。** 実行するprofileは常に `PROFILE=` で明示する。

## Run

```bash
# 計画のみ表示 (GPU不要)
PLAN_ONLY=1 PROFILE=full \
  experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/run_fm_linear_baseline.sh

# GPUが空くまで待ってから実行
PROFILE=smoke GPU_INDEX=0 \
  experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/schedule_after_gpu_free.sh

# unit test (GPU不要)
python experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/test_fm_linear_baseline.py
```

再実行すると、最新の `training-state-*.pt` から `--resume` で再開し、記録済みの
health・sample・FIDはスキップする。現在のstageが終わった時点で安全に止めたい
場合は次を作る。

```bash
touch experiments/7/1_cifar10_cond_linear_fm_baseline/STOP_AFTER_CURRENT_STAGE
```

## Reproduction Conditions

| 項目 | 値 |
| :--- | :--- |
| Dataset | `datasets/cifar10-32x32.zip` (class-conditional) |
| Architecture | DDPM++ (`--arch=ddpmpp`)、55,735,299 params |
| Preconditioning & loss | `--precond=fm` (`FMPrecond` + `FMLoss(path='linear')`) |
| Batch | `--batch=512`、`--batch-gpu=128` (1 GPU、勾配蓄積4回) |
| Optimizer | Adam、`--lr=0.001`、LR ramp-up 10000 kimg |
| EMA | `--ema=0.5` (Mimg半減期) |
| Dropout / Augment | `--dropout=0.13` / `--augment=0.12` |
| Precision | FP32 (`--fp16=0`) |
| Seed | `--seed=0` |
| GPUs | 1 (physical index 0、Quadro RTX 8000) |
| Sampler | `fm-euler`、`--steps=18` (18 NFE) |
| Sample seeds | 0-63 |
| FID reference | `fid-refs/cifar10-32x32.npz` |
| Environment | `/home/kyoukinkin/data/conda-envs/edm` (PyTorch 1.12.1+cu113) |

## Execution Log

### 2026-09-09: 実装とsmoke実行 (Claude Code)

Codexセッション「Find viable paper direction ＋ FMベースライン」の `plan.md` を、
クレジット不足のためCodexの代わりにClaude Codeが実行した。plan.mdの範囲
(実装・段階実験スクリプト・smokeまで) をすべて実施し、pilot/fullは起動していない。

**実装**

- `FMLoss` / `FMPrecond` / `--precond=fm` / `--sampler=fm-euler|fm-heun` /
  `check_sampler_compatibility()` を上表のとおり追加。
- `plan.md` は「既存の未コミットsampler変更を保持する」と書いているが、
  作業開始時のworktreeは `plan.md` 以外clean (`git status` 済み) で、保持すべき
  未コミット変更は無かった。既存のEDM/RK4/PNDM/S-PNDM/AB samplerと既定動作は
  変更していない。

**検証結果 (すべてpass)**

| 検証 | コマンド | 結果 |
| :--- | :--- | :--- |
| 構文 | `python -m compileall generate.py train.py training/loss.py training/networks.py` | OK |
| Unit test | `python .../scripts/test_fm_linear_baseline.py` | 22 tests OK |
| CLI | `train.py --help` / `generate.py --help` | OK (`docs/*.txt` 更新) |
| CLI | `train.py --precond=fm --seed=0 --dry-run` | `FMPrecond` + `FMLoss(path=linear)` を確認 |
| Runner | `PLAN_ONLY=1` × smoke / pilot / full | 全stageの計画を出力、GPU未使用 |

unit testの内訳: linear pathの両端 (`t=0` → noise、`t=1` → data)、`v = y - z` と
数値微分の一致、定数velocity場のEuler/Heun積分と解析解の一致、時間線形な
velocity場での誤差次数 (Heun `b/(2N^2)` < Euler `b/(2N)`、最終区間がEulerである
ことの確認)、単一データ点の最適velocity場 `v = (y* - x)/(1-t)` による
noise → data輸送 (loss側の経路定義とsampler側の時間の向きの整合)、
Euler `N` / Heun `2N-1` のNFE、checkpoint・sampler・option組合せの拒否。

**GPU smoke (`PROFILE=smoke`、`scripts/wait_for_gpu_and_run.sh` 経由)**

- 学習: 50 kimg、seed 0、6分 (`sec/kimg` 6.54、`gpumem` 16.50 GB、`reserved` 18.09 GB)。
  EDM baselineの 6.49-6.52 sec/kimg とほぼ同じ。
- loss: 1.2509 → 1.2528 → 1.2423 → 1.2150 → 1.1762 → 1.1385。
  理論初期値 1.25 から単調に低下しており有限。
- checkpoint再読込: `training-state-000050.pt` (55,735,299 params) と
  `network-snapshot-000050.pkl` を再読込し、パラメータ・loss・全gradientが有限
  (global grad norm 23947.4)。EMAは `prediction_type='velocity'`、`path='linear'`、
  時間範囲 `[0,1]`、`M=1000` を保持し、EDMの `sigma_min`/`sigma_max` を持たない。
  `stats.jsonl` の6 tick分のloss/resource/timingもすべて有限。
- `--resume=training-state-000050.pt --dry-run` が解決することを確認。
- 生成: `fm-euler --steps=18` で seeds 0-63 の64枚を生成。
- 決定性: 同じseedで再生成した64枚がsha256まで完全一致。
- 再実行 (2回目のsmoke): 学習・health・sampleはすべてskipされ、決定性検証と
  checkpoint検証のみ再実行された。このとき比較対象は1回目の実行で生成した
  画像なので、**別プロセス・別実行でも同じseedから同じ画像が得られる**ことを
  確認できている。
- smoke完了後、pilot/fullは自動起動していない。

### 2026-09-09: training-runsを実験フォルダへ集約 (Claude Code)

トップレベルの `training-runs/cifar10-cond-ddpmpp-fm-linear/` (4.4 GB) を
`experiments/7/1_cifar10_cond_linear_fm_baseline/training-runs/cifar10-cond-ddpmpp-fm-linear/`
へ `mv` で移動し、`run_fm_linear_baseline.sh` の `RUN_DIR` 既定値を
`$EXP_ROOT/training-runs/...` に更新した (実験1と同じ「実験フォルダ内に
training-runsを持つ」構成)。プロセス実行中でないこと (`ps`/`nvidia-smi` で確認)
を確かめてから移動し、コピーではなく移動としたのは、既存の
`experiments/1-.../training-runs/` がトップレベルとの重複コピー (68 GB×2) に
なっており、ディスク使用率93%の環境でこれ以上重複を増やしたくなかったため。

移動後、`PLAN_ONLY=1` で新しい `RUN_DIR` を確認し、smokeを再実行して
学習/health/sampleがすべてskipされること、決定性検証・checkpoint検証が
移動後のファイルに対して問題なくPASSすることを確認した。

**実checkpointでの組合せ検証**

| ケース | 結果 |
| :--- | :--- |
| FM checkpoint + 既定 (`edm`) sampler | `Error: --sampler=edm requires an EDM denoiser checkpoint, ...` |
| EDM checkpoint + `fm-euler` | `Error: --sampler=fm-euler requires a flow matching checkpoint trained with --precond=fm, ...` |
| FM checkpoint + `fm-euler --rho=5` | `Error: ... does not use EDM noise levels; remove --rho=5.0` |
| FM checkpoint + `fm-euler --solver=heun` | `Error: ... cannot be combined with ablation options --solver` |
| FM checkpoint + `fm-heun --steps=9` | 成功 (17 NFE、8枚生成) |
| EDM checkpoint + `edm` / `pndm` (回帰確認) | どちらも従来どおり生成成功 |

**smokeのサンプル画像について**

50 kimg は batch 512 で約98 optimizer step、しかもLR ramp-upが10000 kimgなので
実効LRは約5e-6 であり、networkはほぼ初期化状態に近い。そのため生成画像は
空間相関がほとんど無いノイズ状になる (隣接ピクセル相関 0.023)。これは想定内で、
smokeはあくまで数値の有限性・保存/再読込・生成経路・決定性の確認が目的。
画質の判断は pilot (FID-5k) 以降で行う。

### 次にやること (未実行)

1. `PROFILE=pilot` を実行し、1k/3k/10k/20k kimg のFID-5kでEDM baselineと
   学習初期の立ち上がりを比較する。
2. 問題なければ `PROFILE=full` で200k kimgまで進め、最終kimgでFID-50k×3 seed
   範囲を測る。
3. `fm-heun` (35 NFE) と `fm-euler` (18 NFE) のsampler比較は、`experiments/5` の
   sampler実験と同じ形式で別実験として切り出す。
4. Minibatch OT couplingは今回の範囲外。入れる場合は `FMLoss.sample_noise` を
   差し替える。

## Assumptions

- 最初のbaselineは独立couplingの単純linear pathとし、Minibatch OTは範囲外。
- EDMとの比較可能性を優先し、FM固有のハイパーパラメータ探索は行わない。
- 学習は新規初期化し、EDM checkpointからtransferしない。
- GPUはphysical index 0、想定GPU名は Quadro RTX 8000。
- 生成画像、checkpoint、大容量FID一時データはコミット対象外。
