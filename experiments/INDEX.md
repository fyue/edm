# 実験インデックス

`python scripts/build_experiment_index.py` が自動生成する。手で編集しない。
全体の地図と読む順序は [README.md](README.md)、グループごとの問いと結論は各グループの `README.md` を参照する。
概要列は各実験READMEのH1直後の段落（無ければ「目的」で始まる節の先頭段落）を切り詰めたもの。

## 0（[グループREADME](0/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [0-1](0/1_cifar10_cond_edm_10k_retraining/README.md) | Experiment 0-1: CIFAR-10 Conditional EDM Exact 10k Retraining | 修論本番用にCIFAR-10 class-conditional EDM/DDPM++をseed 0から200kまで再学習し、 10k刻みの均一なcheckpoint系列を作る。既存実験で評価した `5.5k, 20k, 120k, 200k` の不均一な選択とは分離し、 `1… | ✓ |
| [0-2](0/2_cifar10_cond_edm_30k_subset/README.md) | Experiment 0-2: CIFAR-10 Conditional EDM on a 30k Subset | CIFAR-10の30,000枚subsetで学習し、データ量が学習進行とFIDに与える影響を調べる。100kまでのFID-50kは50k時点の2.34864が最良で、その後100kでは4.44395へ悪化した（評価結果）。 | ✓ |
| [0-3](0/3_cifar10_cond_edm_post200k_extension/README.md) | 実験0-3：200k以降の継続学習 | 実験0-1の200kから学習率0.001を維持して210kまで継続し、FIDの改善を調べる。 元の実験0-1を読み取り専用の出発点として使い、この実験のtraining-runsへ保存する。 今回は210kで停止する。220k以降への延長は明示的にtargetを指定する。 | ✓ |
| [0-4](0/4_post200k_restart_k1_fid50k_3seeds/README.md) | 実験0-4：200k・210kのRestart K=1効果、FID-50k・3生成seed範囲 | 固定した200k・210kの各モデルで、Restart K=1のFID改善が生成seed範囲を変えても 再現するかを確認する。学習は行わず、実験0-3のモデル・コード・成果物は読み取りのみとする。 3 seedは3回の再学習ではなく、固定モデルからの3つの生成seed範囲を意味… | ✓ |
| [0-5](0/5_cifar10_cond_edm_10k_subset/README.md) | Experiment 0-5: CIFAR-10 Conditional EDM on a 10k Subset | 0-2と同じ学習・評価条件を使い、学習サブセットを30,000枚から10,000枚へ変更する。 seed 0から新規学習し、30,000 kimg（30 Mimg）まで学習する。 ここでサブセットの10kは画像枚数、学習量の30kはkimg単位を表す。 | ✓ |

## 1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18

- [EDM CIFAR-10 本番学習メモ](1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/README.md)：CIFAR-10 class-conditional EDM/DDPM++ を、論文の **FID 1.79** にできるだけ近い条件で再現する。

## 2

- [Experiment 2: 120k/200k conditional generation check](2/README.md)：本番学習済みの CIFAR-10 class-conditional EDM checkpoint について、`generate.py` の `--class` 指定が問題なく効くかを確認する。

## 3（[グループREADME](3/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [3-1](3/1_120k-160k-lr-ema-sweep/README.md) | Experiment 3: 120kからのlr/EMA分岐実験 | Experiment 1では、CIFAR-10 class-conditional DDPM++ EDM の本番学習で 120k のFIDが最も良く、その後 160k/200k で悪化しました。 | ✓ |
| [3-2](3/2_160k-200k-lr5e-4-ema0.5/README.md) | Experiment 3 continuation: lr=0.0005, EMA=0.5, 160k -> 200k | `120k-160k-lr-ema-sweep` では、120k -> 160k の分岐実験で `lr=0.0005, EMA=0.5` が最も良いFIDを示しました。 | ✓ |
| [3-3](3/3_160k-170k-lr2e-4-ema0.5/README.md) | Experiment 3 continuation: lr=0.0002, EMA=0.5, 160k -> 170k | `120k-160k-lr-ema-sweep` で最も良いFIDを示した 160k checkpoint から、 EMA half-lifeは `0.5` のまま learning rate を `0.0002` に下げて 170k まで継続します。 170k時点のFIDを確… | ✓ |
| [3-4](3/4_fid50k_170k_lr5e-4_ema0.5/README.md) | 実験3-4: lr=0.0005, EMA=0.5 の170k FID-50k | `160k-200k-lr5e-4-ema0.5` の学習途中に保存された170k checkpointについて、 FID-50kを計算します。学習は行わず、保存済みsnapshotから画像生成とFID計算だけを実行します。 | ✓ |
| [3-5](3/5_170k-200k-lr2e-4-ema0.5/README.md) | Experiment 3_5: lr=0.0002, EMA=0.5, 170k -> 180k + 175k再評価 | 実験3_3では、`lr=0.0002, EMA=0.5` で160kから170kまで継続し、170k時点でFIDが改善しました。 | ✓ |
| [3-6](3/6_170k-180k-lr1e-4-ema0.5/README.md) | Experiment 3_6: lr=0.0001, EMA=0.5, 170k -> 180k | 実験3_3で改善した170k checkpointを起点に、learning rateを `0.0001` に下げて 175k、180kまで段階的に継続して評価します。加えて、起点の170kも同じ条件で 再評価し、ベースラインの再現性を確認します。実験3_5（`lr=0.000… | ✓ |
| [3-7](3/7_170k-180k-lr5e-5-ema0.5/README.md) | Experiment 3_7: lr=0.00005, EMA=0.5, 170k -> 180k | 実験3_3で改善した170k checkpointを起点に、learning rateを `0.00005` に下げて 175k、180kまで段階的に継続して評価します。加えて、起点の170kも同じ条件で 再評価し、ベースラインの再現性を確認します。実験3_6（`lr=0.00… | ✓ |
| [3-8](3/8_fid50k_167526_165018_lr2e-4_ema0.5/README.md) | Experiment 3_8: 167526 / 165018 snapshots の3 seed評価 | 実験3_3 `lr=0.0002, EMA=0.5, 160k -> 170k` の保存済みsnapshotから、次の2点をFID-50kで3 seed範囲評価します。 | ✓ |
| [3-9](3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/README.md) | Experiment 3-9: lr=2e-4 / lr=1e-4 の 172509 / 175018 snapshot 3 seed FID評価 | 次の4条件について、FID-50kを3 seed範囲で評価します。学習は行わず、保存済みsnapshotを参照します。 | ✓ |
| [3-10](3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/README.md) | Experiment 3-10: 160k FID~2.05 checkpoint -> 170k, lr=2e-4 | Experiment 3-1 の `lr1e-3-ema0.5-repro` 分岐で得られた、160k時点の3 seed平均FIDがおよそ `2.05` のcheckpointを起点に、`lr=0.0002`, `EMA=0.5` で170kまで継続学習する。 | ✓ |
| [3-11](3/11_fid50k_around170k_lr5e-4_ema0.5/README.md) | Experiment 3-11: lr=5e-4, EMA=0.5, 170k周辺4 checkpointの3 seed評価 | `lr=0.0005`, `EMA=0.5` 条件で生成済みの checkpoint から、170k 近傍の4点を FID-50k の3 seed範囲で評価する。 | ✓ |
| [3-12](3/12_170k-to-180k-lr2e-4-ema0.5/README.md) | Experiment 3-12: Experiment 3-10 170k checkpoint -> 180k, lr=2e-4 | Experiment 3-10 で得られた 170k checkpoint を起点に、同じ学習条件 `lr=0.0002`, `EMA=0.5` で180kまで継続学習する。 | ✓ |
| [3-13](3/13_170k-to-180k-lr1e-4-ema0.5/README.md) | Experiment 3-13: best 170k checkpoint -> 180k, lr=1e-4 | Experiment 3-10 で得られた最良の 170k checkpoint を起点に、`lr=0.0001`, `EMA=0.5` で180kまで継続学習する。 | ✓ |
| [3-14](3/14_160k-fid2.05-to-175k-lr15e-4-ema0.5/README.md) | Experiment 3-14: 160k FID 2.05 checkpoint -> 175k, lr=15e-4 | Experiment 3-1 の 160k checkpoint のうち、3 seed FID平均が約 `2.05` の枝を起点に、`lr=0.0015`, `EMA=0.5` で 175k まで継続学習する。 | ✓ |
| [3-15](3/15_160k-fid2.05-to-170k-lr15e-4-ema0.5/README.md) | Experiment 3-15: 160k FID 2.05 checkpoint -> 170k, lr=15e-4 | Experiment 3-1 の 160k checkpoint のうち、3 seed FID平均が約 `2.05` の枝を起点に、`lr=0.0015`, `EMA=0.5` で 170k まで継続学習する。 | ✓ |
| [3-16](3/16_160k-fid2.05-to-170k-lr15e-5-ema0.5/README.md) | Experiment 3-16: 160k FID 2.05 checkpoint -> 170k, lr=15e-5 | Experiment 3-1 の 160k checkpoint のうち、3 seed FID平均が約 `2.05` の枝を起点に、`lr=0.00015`, `EMA=0.5` で 170k まで継続学習する。 | ✓ |
| [3-17](3/17_170k-to-175k-same-as-3-16-lr15e-5-ema0.5/README.md) | Experiment 3-17: 3-16 170k checkpoint -> 175k, same setting | Experiment 3-16 の 170k checkpoint を起点に、3-16 と同じ `lr=0.00015`, `EMA=0.5` で 175k まで継続学習する。 | ✓ |
| [3-18](3/18_160k-fid2.05-to-175k-lr1e-4-ema0.5/README.md) | Experiment 3-18: 160k FID 2.05 checkpoint -> 175k, lr=1e-4 | Experiment 3-1 の 160k checkpoint のうち、3 seed FID平均が約 `2.05` の枝を起点に、`lr=0.0001`, `EMA=0.5` で 175k まで継続学習する。 | ✓ |
| [3-19](3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/README.md) | Experiment 3-19: 175k FID 1.89 checkpoint -> 190k, lr=1e-4 | Experiment 3-18 の 175k checkpoint のうち、3 seed FID平均が約 `1.89` の枝を起点に、3-18 と同じ `lr=0.0001`, `EMA=0.5` で 190k まで継続学習する。 | ✓ |
| [3-20](3/20_175k-fid1.89-to-190k-lr5e-5-ema0.5/README.md) | Experiment 3-20: 175k FID 1.89 checkpoint -> 190k, lr=5e-5 | Experiment 3-18 の 175k checkpoint のうち、3 seed FID平均が約 `1.89` の枝を起点に、`lr=0.00005`, `EMA=0.5` で 190k まで継続学習する。 | ✓ |

資料（実験READMEなし）：`3/report/`

## 4（[グループREADME](4/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [4/pndm](4/pndm/README.md) | Experiment 4 PNDM: checkpoint-wise S_churn sensitivity | Experiment 4 の EDM sampler に対する `S_churn` 感度実験を、EDM座標系へ移植した PNDM sampler（`generate.py --sampler=pndm`）で再現する。 |  |
| [4/stage0-1](4/stage0/1_3x2/README.md) | Experiment 4 Stage 0 3x2: S_churn sensitivity smoke test | CIFAR-10 class-conditional EDM の学習段階ごとに、`S_churn` と生成品質の関係を低コストに確認する。 | ✓ |
| [4/stage0-2](4/stage0/2_4check_4s_churn/README.md) | Experiment 4 Stage 0 4check_4s_churn | CIFAR-10 class-conditional EDM の学習段階ごとに、`S_churn` と生成品質の関係を低コストに確認する。 | ✓ |
| [4/stage0-3](4/stage0/3_fine_s_churn_under_clip/README.md) | Experiment 4 Stage 0 Third: Fine S_churn Sweep Under Clip | Stage 0 の3番目の実験として、`S_churn=20` と `S_churn=40` が同じ結果になった理由を踏まえ、クリップ閾値未満の `S_churn` を細かく調べる。 | ✓ |
| [4/stage1-1](4/stage1/1_fid5k_selected_churn/README.md) | Experiment 4 Stage 1 First: Selected S_churn FID-5k | Stage 0 の FID-1k で見えた傾向を、少し信頼度の高い `FID-5k` で確認する。 | ✓ |
| [4/stage1-2](4/stage1/2_fid5k_churn_6_7/README.md) | Experiment 4 Stage 1 Second: S_churn=6,7 FID-5k Extension | Stage 1 実験1の `S_churn=0-5` を変更せず、`S_churn=6,7` を追加してFID-5kがさらに下がるか確認する。 | ✓ |
| [4/stage2-1](4/stage2/1_fid50k_selected_churn/README.md) | Experiment 4 Stage 2 First: Selected S_churn FID-50k | Stage 1 の `FID-5k` で見えた傾向を、正式評価に近い `FID-50k` で確認する。 | ✓ |
| [4/stage2-2](4/stage2/2_fid50k_extra_seeds/README.md) | Experiment 4 Stage 2 Second: Extra Seeds for Representative FID-50k Pairs | Stage 2 experiment 1 の代表ペアについて、追加の2 seed範囲を評価し、3 seed範囲で `mean ± std` を出す。 | ✓ |
| [4/stage2-3](4/stage2/3_fid50k_churn_6_7/README.md) | Experiment 4 Stage 2 Third: S_churn=6,7 FID-50k Extension | Stage 2実験1を変更せず、同じcheckpoint・seed・sampler条件で `S_churn=6,7` のFID-50kを追加評価し、既存結果と結合した結果画像を作る。 | ✓ |

資料（実験READMEなし）：`4/docs/`、`4/report/`

## 5（[グループREADME](5/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [5-1](5/1_pndm_fid50k_3seeds/README.md) | Experiment 5-1: PNDM Sampler FID-50k over 3 Seed Ranges | `generate.py --sampler=pndm` の FID-50k を、3つの seed 範囲で評価する。 | ✓ |
| [5-2](5/2_spndm_fid50k_3seeds/README.md) | Experiment 5-2: S-PNDM Sampler FID-50k over 3 Seed Ranges | `generate.py --sampler=spndm` の FID-50k を、3つの seed 範囲で評価する。 | ✓ |
| [5-3](5/3_rk4_fid50k_3seeds/README.md) | Experiment 5-3: RK4 Sampler FID-50k over 3 Seed Ranges | `generate.py --sampler=rk4` の FID-50k を、3つの seed 範囲で評価する。 | ✓ |
| [5-4](5/4_ab_fid50k_3seeds/README.md) | Experiment 5-4: Adams-Bashforth Multiple-Step Sampler FID-50k over 3 Seed Ranges | `generate.py --sampler=ab` の FID-50k を、3つの seed 範囲で評価する。 | ✓ |
| [5-5](5/5_pndm_35nfe_fid50k_3seeds/README.md) | Experiment 5-5: PNDM Sampler 35NFE FID-50k over 3 Seed Ranges | `generate.py --sampler=pndm` の35NFE条件での FID-50k を、3つの seed 範囲で評価する。 | ✓ |
| [5-6](5/6_pndm_linear_fid50k_3seeds/README.md) | Experiment 5-6: PNDM Fixed AB4 with Linear Sigma Schedule FID-50k over 3 Seed Ranges | `generate.py --sampler=pndm-linear` の FID-50k を、3つの seed 範囲で評価する。 | ✓ |
| [5-7](5/7_ab_35nfe_fid50k_3seeds/README.md) | Experiment 5-7: No-RK Warmup AB 35NFE FID-50k over 3 Seed Ranges | `generate.py --sampler=ab`を35 steps / 35NFEで実行し、FID-50kを3つの seed範囲で評価する。 | ✓ |
| [5-8](5/8_pndm_no_rk_warmup_fid50k_3seeds/README.md) | Experiment 5-8: PNDM without RK Warmup | 実験5-5のPNDMから最初の3区間のRK4 warmupだけを除き、FID-50kを3つの seed範囲で評価する。no-RK版はEuler, AB2, AB3で履歴を作り、その後は 実験5-5と同じ固定係数AB4を使う。 | ✓ |
| [5-9](5/9_pndm_no_rk_35nfe_fid50k_3seeds/README.md) | Experiment 5-9: PNDM no-RK 35NFE FID-50k over 3 Seed Ranges | 実験5-8の `pndm-no-rk` を `steps=35`, `NFE=35` に増やし、FID-50kを3つのseed範囲で評価する。 | ✓ |
| [5-10](5/10_spndm_35nfe_fid50k_3seeds/README.md) | Experiment 5-10: S-PNDM Sampler 35NFE FID-50k over 3 Seed Ranges | 実験5-2を基準に、`generate.py --sampler=spndm` の35NFE条件でFID-50kを 3つのseed範囲について評価する。 | ✓ |
| [5-11](5/11_pndm_no_rk_18nfe_fid50k_3seeds/README.md) | Experiment 5-11: PNDM no-RK 18NFE FID-50k over 3 Seed Ranges | 固定係数Adams-Bashforthを使う `pndm-no-rk` を18 steps / 18 NFEで評価する。 実験5-4の可変係数ABとsteps、NFE、sigma schedule、warmup、seed範囲を揃え、 非等間隔のEDM rho scheduleに… | ✓ |
| [5-12](5/12_ab_26nfe_fid50k_3seeds/README.md) | Experiment 5-12: Variable-Coefficient AB 26NFE FID-50k over 3 Seed Ranges | EDM rho scheduleに合わせた可変係数Adams-Bashforth sampler `ab` を 26 steps / 26 NFEで評価する。実験5-8の固定係数PNDM no-RKとsteps、NFE、 sigma schedule、warmup、seed範囲… | ✓ |
| [5-13](5/13_trajectory_distance_5-4_vs_5-11/README.md) | Experiment 5-13: Trajectory Distance for 5-4 vs 5-11 | 実験5-4（不等間隔に合わせた可変係数AB）と実験5-11（等間隔用の固定係数 PNDM no-RK）が、同じ学習済みdenoiserのODEに対する高精度参照軌道からどれだけ 離れるかを測る。 | ✓ |

資料（実験READMEなし）：[SAMPLER_EXPERIMENT_TEMPLATE.md](5/SAMPLER_EXPERIMENT_TEMPLATE.md)、`5/report/`

## 6（[グループREADME](6/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [6-1](6/1_edm_forward_norm_density/README.md) | Experiment 6-1: EDM Forward-Noising Norm Density | PNDM論文 Figure 2 のように、前向きノイズ付加後のデータのノルム分布を可視化する。 | ✓ |

## 7（[グループREADME](7/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [7-1](7/1_cifar10_cond_linear_fm_baseline/README.md) | Experiment 7-1: CIFAR-10 Class-Conditional Linear Flow Matching Baseline | CIFAR-10 class-conditionalを対象に、EDMと同じDDPM++構造・学習設定を使う linear-path flow matching (FM) baselineを追加する。EDMとFMの違いを 「経路と学習目標」だけに絞り、architecture・b… | ✓ |
| [7-2](7/2_cifar10_cond_linear_fm_pilot/README.md) | Experiment 7-2: CIFAR-10 Conditional Linear FM Pilot | Experiment 7-1で動作確認したlinear-path Flow Matching baselineを、修論で傾向を 判断できる20,000 kimgまで段階的に学習する。1,000 / 3,000 / 10,000 / 20,000 kimgで学習状態、固定seed… | ✓ |
| [7-3](7/3_fm_solver_nfe_and_edm_comparison/README.md) | Experiment 7-3: FM Solver/NFE Sweep and EDM Comparison | Experiment 7-2で20,000 kimgまで学習したlinear FMについて、生成品質の差が 学習誤差ではなく数値積分誤差から来ている可能性を調べる。同じFM velocity場を Euler/Heunで積分し、同一NFEでFID-5kを比較する。さらに、学習条件… | ✓ |
| [7-4](7/4_fm_edm_fid50k_confirmation/README.md) | Experiment 7-4: FM/EDM FID-50k Confirmation | Experiment 7-3のFID-5kで得た主要傾向を、標準的な50,000生成画像のFIDで確認する。 20,000 kimgまで学習した同一規模のCIFAR-10 conditionalモデルを使い、次の2点を検証する。 | ✓ |
| [7-5](7/5_fm_edm_trajectory_error/README.md) | Experiment 7-5: FM/EDM Trajectory and Integration Error | Experiment 7-3と7-4では、FMはNFEを増やすと改善する一方、同じNFE 35では EDMよりFIDが悪かった。本実験では、この差を次の2要素に分けて調べる。 | ✓ |

## 8（[グループREADME](8/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [8-1](8/1_restart_sampling_checkpoint_sweep/README.md) | Experiment 8-1: Restart Sampling Checkpoint Sweep | CIFAR-10 class-conditional EDMの学習途中checkpointに対し、Restart Samplingが通常の ODE（EDM schedule + Heun）より生成品質を改善するか調べる。特に、学習不足なmodel ほどRestartの改善幅が大… | ✓ |
| [8-2](8/2_restart_k5_fid5k_confirmation/README.md) | Experiment 8-2: Selected Restart K FID-5k Confirmation | Experiment 8-1のFID-1k screeningで観測した「学習不足modelほどRestartの改善幅が 大きい」という傾向を、FID-5kで再確認する。 | ✓ |
| [8-3](8/3_restart_k5_fid50k_3seeds/README.md) | Experiment 8-3: Restart K=5 FID-50k over 3 Seed Ranges | Experiment 8-2のFID-5kで確認した「Restartは学習不足checkpointでは改善し、学習後期 では悪化する」という傾向を、FID-50k × 3独立seed範囲で正式評価する。 | ✓ |
| [8-4](8/4_restart_k_sweep_10k_interval/README.md) | Experiment 8-4: Restart K Sweep at 10k Checkpoint Intervals | 厳密な10k刻み再学習系列の利用可能な全checkpointで`K=0,1,...,8`を走査し、 FIDを最小にするKが学習進行に伴って変化するか調べる。checkpointは実行時に検出し、 K=0～8の条件TSVを自動生成する。checkpoint追加時にコードやTSVを… | ✓ |
| [8-6](8/6_exact_10k_restart_k5_fid50k_1seed/README.md) | Experiment 8-6: Progressive Exact-10k Restart K=5 FID Evaluation | Experiment 0-1で同一学習系列から保存した`10k, 20k, ..., 120k`の12 checkpointを、 Experiment 8-3と同じsampling条件で評価する。FID画像数は一度に50kまで生成せず、 `5k → 25k → 50k`の順に段… | ✓ |
| [8-7](8/7_trained_restart_k1_fid50k_1seed/README.md) | Experiment 8-7: Trained Restart K=1 vs. Equal-NFE ODE, FID-50k | 十分学習した2 checkpointについて、Restart K=1と同計算量の通常ODEを比較する。 各条件のFID-50kをseed範囲`0-49999`の1組で評価する。「1 seed」は1枚の生成ではなく、 50,000個の生成seedからなる1範囲を意味する。seed… | ✓ |
| [8-8](8/8_restart_tmin_comparison/README.md) | 実験8-8：Restart t_min 比較（K=5） | EDM条件付きCIFAR-10の20k・40k・60kと、実験3-10の最良170kチェックポイントで、Restart区間 `(0.06, 0.30)` と `(0.14, 0.30)` を比較する。同じ画像集合をFID-5k→25k→50kへ段階的に拡張する。8条件・全24件… | ✓ |
| [8-10](8/10_restart_fixed_nfe55_k_sweep/README.md) | 実験8-10：NFE=55固定で学習量とKを比較 | 実験0-1のCIFAR-10 class-conditional EDM（DDPM++、学習seed=0）を使い、学習量ごとに、固定計算量で主経路とRestartへの計算配分を比較する。K=5は今回の生成・集計対象から除外する。実験8-4のK=5結果の取り込みは後続作業とする。 | ✓ |
| [8-11](8/11_restart_fixed_main18_tmin014_k_sweep/README.md) | 実験8-11：N_main=18・t_min=0.14固定のK sweep | 実験0-1のCIFAR-10 class-conditional EDM（DDPM++、学習seed=0）について、主経路を18 stepsに固定し、Restart回数Kと学習量の関係を比較する。計算量はKに応じて変化する。実験8-10のNFE固定比較とは異なる条件である。 | ✓ |
| [8-12](8/12_restart_fixed_nfe55_tmin014_k_sweep/README.md) | 実験8-12：NFE=55・t_min=0.14固定のK sweep | 実験0-1のCIFAR-10 class-conditional EDM（DDPM++、学習seed=0）を使い、固定計算量で主経路とRestartへの計算配分を比較する。実験8-10のt_min=0.06固定比較、実験8-11のN_main=18固定比較とは条件が異なる。両P… | ✓ |
| [8-13](8/13_restart_optimal_k_fid50k_3seeds/README.md) | 実験8-13：最適KとK=0のFID-50k・3シード比較 | 8-11で選択した最適Kを固定し、固定モデルにおける生成シード間のFID変動とpaired改善幅を確認する。学習seedは0のままであり、3回の再学習ではない。 | ✓ |
| [8-14](8/14_restart_subset30k_k_sweep/README.md) | 実験8-14：30,000枚学習の20k〜100kにおけるRestart K比較 | 実験0-2（CIFAR-10 class-conditional、DDPM++/EDM、学習seed=0、subset seed=0、30,000枚）の保存済みcheckpointを使う。追加学習は行わない。ベースラインFIDが悪いほど同じKでRestart改善幅が大きいかを調… |  |

資料（実験READMEなし）：`8/report/`

## reports

資料（実験READMEなし）：[2026-09-17_additional_experiments_report.md](reports/2026-09-17_additional_experiments_report.md)、[2026-10-02.plan.md](reports/2026-10-02.plan.md)、`reports/assets/`、`reports/scripts/`

## slack

- [加藤先生とのRestart研究の議論・必要図](slack/README.md)：2026-11-03 JST時点の整理。まず概要と合意事項、次に必要図一覧と対応事項を参照する。

## 直下の資料

- [spring_semester_report.md](spring_semester_report.md)
