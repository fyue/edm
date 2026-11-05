# 実験インデックス

`python scripts/build_experiment_index.py` が自動生成する。手で編集しない。
全体の地図と読む順序は [README.md](README.md)、グループごとの問いと結論は各グループの `README.md` を参照する。
概要列は各実験READMEのH1直後の段落（無ければ「目的」で始まる節の先頭段落）を切り詰めたもの。
results列は `results/` または `runs/*/results/` にREADME以外の出力があれば✓。完了を意味しない。

## 0（[グループREADME](0/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [0-1](0/1_cifar10_cond_edm_10k_retraining/README.md) | Experiment 0-1: CIFAR-10 Conditional EDM Exact 10k Retraining | 全データ・学習seed 0で0→200kを1プロセスで連続学習し、10k刻み20点の均一なcheckpoint系列を作る。完了済みで、8-4・8-6・8-10〜8-13の評価対象（本実験ではFIDを評価しない）。 | ✓ |
| [0-2](0/2_cifar10_cond_edm_30k_subset/README.md) | Experiment 0-2: CIFAR-10 Conditional EDM on a 30k Subset | CIFAR-10の30,000枚subsetで学習し、データ量が学習進行とFIDに与える影響を調べる。100kまでのFID-50kは50k時点の2.34864が最良で、その後100kでは4.44395へ悪化した（評価結果）。 | ✓ |
| [0-3](0/3_cifar10_cond_edm_post200k_extension/README.md) | 実験0-3：200k以降の継続学習 | 実験0-1の200kから学習率0.001のまま210k・220kまで継続学習し、FIDが改善するかを調べる。FID-50k（seed 0-49999）ではK=0が200k 1.80678→210k 1.83710→220k 2.00929と悪化し、K=1も同様だった。 | ✓ |
| [0-4](0/4_post200k_restart_k1_fid50k_3seeds/README.md) | 実験0-4：200k・210kのRestart K=1効果、FID-50k・3生成seed範囲 | 200k・210kでRestart K=1のFID改善が3つの生成seed範囲で再現するかをFID-50kで確かめる。このPCには初回範囲の4評価のみ（K=1の改善は200k +0.01343、210k +0.01809）で、3範囲の結果は元PCと要照合。 | ✓ |
| [0-5](0/5_cifar10_cond_edm_10k_subset/README.md) | Experiment 0-5: CIFAR-10 Conditional EDM on a 10k Subset | 0-2と同じ条件で学習データを10,000枚に減らし、30k（kimg）まで学習してデータ量の影響を調べる。30kまで完了し、FID-50kは10k 4.63680、20k 2.82997（最良）、30k 3.77162だった。 | ✓ |

## 1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18

- [EDM CIFAR-10 本番学習メモ](1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/README.md)：CIFAR-10 class-conditional EDM/DDPM++ を、論文の **FID 1.79** にできるだけ近い条件で再現する。

## 2

- [Experiment 2: 120k/200k conditional generation check](2/README.md)：本番学習済みの CIFAR-10 class-conditional EDM checkpoint について、`generate.py` の `--class` 指定が問題なく効くかを確認する。

## 3（[グループREADME](3/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [3-1](3/1_120k-160k-lr-ema-sweep/README.md) | Experiment 3-1: 120kからのlr/EMA分岐実験 | 基準学習の120kから160kまで、lrとEMA half-lifeを変えた3分岐を学習する。160kのFID-50k（3 seed平均）はlr=5e-4で2.010、元設定（lr=1e-3）で2.050、EMA=1.0で2.162だった。 | ✓ |
| [3-2](3/2_160k-200k-lr5e-4-ema0.5/README.md) | Experiment 3 continuation: lr=0.0005, EMA=0.5, 160k -> 200k | 3-1のlr=5e-4の160kから、同じ設定で200kまで継続する。FID-50k（3 seed平均）は180kで2.032、200kで2.292と悪化した。 | ✓ |
| [3-3](3/3_160k-170k-lr2e-4-ema0.5/README.md) | Experiment 3 continuation: lr=0.0002, EMA=0.5, 160k -> 170k | 3-1のlr=5e-4の160kから、lr=2e-4に下げて170kまで継続する。170kのFID-50k（3 seed平均）は1.939で、起点の2.010より改善した。 | ✓ |
| [3-4](3/4_fid50k_170k_lr5e-4_ema0.5/README.md) | 実験3-4: lr=0.0005, EMA=0.5 の170k FID-50k | 3-2（lr=5e-4）の学習途中の170k（170035）をFID-50k×3 seed範囲で評価する（学習なし）。3 seed平均は2.011だった。 | ✓ |
| [3-5](3/5_170k-200k-lr2e-4-ema0.5/README.md) | Experiment 3_5: lr=0.0002, EMA=0.5, 170k -> 180k + 175k再評価 | 3-3の170kから同じlr=2e-4で継続する（200kの予定を180kで打ち切った）。FID-50k（3 seed平均）は175kで2.274、180kで2.298と、170kの1.939から悪化した。 | ✓ |
| [3-6](3/6_170k-180k-lr1e-4-ema0.5/README.md) | Experiment 3_6: lr=0.0001, EMA=0.5, 170k -> 180k | 3-3の170kからlr=1e-4で180kまで継続する。FID-50k（3 seed平均）は175kで2.264、180kで2.662と悪化した。170kの再評価は3-3の値と完全に一致した。 | ✓ |
| [3-7](3/7_170k-180k-lr5e-5-ema0.5/README.md) | Experiment 3_7: lr=0.00005, EMA=0.5, 170k -> 180k | 3-3の170kからlr=5e-5で180kまで継続する。FID-50k（3 seed平均）は175kで2.228、180kで2.529と悪化した。170kの再評価は3-3の値と完全に一致した。 | ✓ |
| [3-8](3/8_fid50k_167526_165018_lr2e-4_ema0.5/README.md) | Experiment 3_8: 167526 / 165018 snapshots の3 seed評価 | 3-3（lr=2e-4）の途中snapshotの165kと167.5kをFID-50k×3 seed範囲で評価する（学習なし）。3 seed平均は165kで1.976、167.5kで2.014で、170kの1.939には届かない。 | ✓ |
| [3-9](3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5/README.md) | Experiment 3-9: lr=2e-4 / lr=1e-4 の 172509 / 175018 snapshot 3 seed FID評価 | 3-5（lr=2e-4）と3-6（lr=1e-4）の172.5k・175kをFID-50k×3 seed範囲で評価する（学習なし）。3 seed平均は2.254〜2.302で、どれも170kより悪い。 | ✓ |
| [3-10](3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/README.md) | Experiment 3-10: 160k FID~2.05 checkpoint -> 170k, lr=2e-4 | 3-1の元設定の160k（FID 2.050）からlr=2e-4で170kまで継続する。170kのFID-50k（3 seed平均）は1.851で、実験3の最良。8-7・8-8の「実験3のbest」はこのsnapshot。 | ✓ |
| [3-11](3/11_fid50k_around170k_lr5e-4_ema0.5/README.md) | Experiment 3-11: lr=5e-4, EMA=0.5, 170k周辺4 checkpointの3 seed評価 | 3-2（lr=5e-4）の170k付近の4 checkpoint（165k〜172.5k）をFID-50k×3 seed範囲で評価する（学習なし）。3 seed平均は2.011〜2.038でほぼ横ばいだった。 | ✓ |
| [3-12](3/12_170k-to-180k-lr2e-4-ema0.5/README.md) | Experiment 3-12: Experiment 3-10 170k checkpoint -> 180k, lr=2e-4 | 3-10の最良170k（1.851）から同じlr=2e-4で180kまで継続する。172.5k〜180kのFID-50k（3 seed平均）は2.038〜2.140で、すべて170kより悪化した。 | ✓ |
| [3-13](3/13_170k-to-180k-lr1e-4-ema0.5/README.md) | Experiment 3-13: best 170k checkpoint -> 180k, lr=1e-4 | 3-10の最良170k（1.851）からlr=1e-4で180kまで継続する。172.5k〜180kのFID-50k（3 seed平均）は2.026〜2.135で、すべて170kより悪化した。 | ✓ |
| [3-14](3/14_160k-fid2.05-to-175k-lr15e-4-ema0.5/README.md) | Experiment 3-14: 160k FID 2.05 checkpoint -> 175k, lr=15e-4 | 3-1の元設定の160k（2.050）からlr=1.5e-3に上げて175kまで継続する。FID-50k（3 seed平均）は2.306〜2.541で、すべて起点より悪化した。 | ✓ |
| [3-15](3/15_160k-fid2.05-to-170k-lr15e-4-ema0.5/README.md) | Experiment 3-15: 160k FID 2.05 checkpoint -> 170k, lr=15e-4 | 3-1の元設定の160kからlr=1.5e-3で170kまで継続する。学習は170kまで完了したが、FID評価は2026-07-11に開始直後で止まり、未評価。同じ起点・lrの結果は3-14を参照。 | ✓ |
| [3-16](3/16_160k-fid2.05-to-170k-lr15e-5-ema0.5/README.md) | Experiment 3-16: 160k FID 2.05 checkpoint -> 170k, lr=15e-5 | 3-1の元設定の160k（2.050）からlr=1.5e-4で170kまで継続する。170kのFID-50k（3 seed平均）は1.854で、3-10（lr=2e-4）の1.851とほぼ同じ。 | ✓ |
| [3-17](3/17_170k-to-175k-same-as-3-16-lr15e-5-ema0.5/README.md) | Experiment 3-17: 3-16 170k checkpoint -> 175k, same setting | 3-16の170k（1.854）から同じlr=1.5e-4で175kまで継続する。FID-50k（3 seed平均）は172.5kで2.009、175kで2.014と悪化した。 | ✓ |
| [3-18](3/18_160k-fid2.05-to-175k-lr1e-4-ema0.5/README.md) | Experiment 3-18: 160k FID 2.05 checkpoint -> 175k, lr=1e-4 | 3-1の元設定の160k（2.050）からlr=1e-4で175kまで継続する。FID-50k（3 seed平均）は170kで1.858（3-10とほぼ同じ）、175kで1.890だった。 | ✓ |
| [3-19](3/19_175k-fid1.89-to-190k-lr1e-4-ema0.5/README.md) | Experiment 3-19: 175k FID 1.89 checkpoint -> 190k, lr=1e-4 | 3-18の175k（1.890）から同じlr=1e-4で190kまで継続する。FID-50k（3 seed平均）の最良は185kの1.910で、190kでは2.027と悪化した。 | ✓ |
| [3-20](3/20_175k-fid1.89-to-190k-lr5e-5-ema0.5/README.md) | Experiment 3-20: 175k FID 1.89 checkpoint -> 190k, lr=5e-5 | 3-18の175k（1.890）からlr=5e-5で190kまで継続する。FID-50k（3 seed平均）の最良は185kの1.903で、190kでは2.010と悪化した。 | ✓ |

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
| [5-1](5/1_pndm_fid50k_3seeds/README.md) | Experiment 5-1: PNDM Sampler FID-50k over 3 Seed Ranges | 公式pretrainedで、EDM座標系へ移植したPNDM（RK4 warmup＋固定係数AB4）を18 steps／27 NFEで評価する。FID-50k（3 seed平均）は1.99152で、EDM Heun（35 NFE）の1.83906に届かない。 | ✓ |
| [5-2](5/2_spndm_fid50k_3seeds/README.md) | Experiment 5-2: S-PNDM Sampler FID-50k over 3 Seed Ranges | 公式pretrainedで、S-PNDM（初回だけHeun補正）を18 steps／19 NFEで評価する。FID-50k（3 seed平均）は3.14766だった。 | ✓ |
| [5-3](5/3_rk4_fid50k_3seeds/README.md) | Experiment 5-3: RK4 Sampler FID-50k over 3 Seed Ranges | 公式pretrainedで、RK4を18 steps／69 NFEで評価する。FID-50k（3 seed平均）は1.92574で、NFEが約2倍でもEDM Heun（35 NFE）の1.83906より悪い。 | ✓ |
| [5-4](5/4_ab_fid50k_3seeds/README.md) | Experiment 5-4: Adams-Bashforth Multiple-Step Sampler FID-50k over 3 Seed Ranges | 公式pretrainedで、rho scheduleに合わせた可変係数Adams-Bashforth（RK warmupなし）を18 steps／18 NFEで評価する。FID-50k（3 seed平均）は3.57748だった。 | ✓ |
| [5-5](5/5_pndm_35nfe_fid50k_3seeds/README.md) | Experiment 5-5: PNDM Sampler 35NFE FID-50k over 3 Seed Ranges | 5-1のPNDMをsteps=26にしてEDM Heunと同じ35 NFEで評価する。FID-50k（3 seed平均）は1.91047で、EDM Heunの1.83906に届かない。 | ✓ |
| [5-6](5/6_pndm_linear_fid50k_3seeds/README.md) | Experiment 5-6: PNDM Fixed AB4 with Linear Sigma Schedule FID-50k over 3 Seed Ranges | 5-1のPNDMのsigma scheduleだけを等間隔（linear）に変える（18 steps／27 NFE）。FID-50k（3 seed平均）は98.1024で失敗。最後の1 stepでσ=4.44から0へ飛ぶなど、低ノイズ側のstepが足りないためと考えられる。 | ✓ |
| [5-7](5/7_ab_35nfe_fid50k_3seeds/README.md) | Experiment 5-7: No-RK Warmup AB 35NFE FID-50k over 3 Seed Ranges | 5-4の可変係数ABを35 steps／35 NFEに増やして評価する。FID-50k（3 seed平均）は2.12443で、18 NFEの3.57748から改善したが、同じ35 NFEのPNDM（1.91047）やEDM Heun（1.83906）より悪い。 | ✓ |
| [5-8](5/8_pndm_no_rk_warmup_fid50k_3seeds/README.md) | Experiment 5-8: PNDM without RK Warmup | 5-5のPNDMからRK4 warmupだけを除く（固定係数AB、26 steps／26 NFE）。FID-50k（3 seed平均）は1.91000で、35 NFEの5-5（1.91047）と同等だった。 | ✓ |
| [5-9](5/9_pndm_no_rk_35nfe_fid50k_3seeds/README.md) | Experiment 5-9: PNDM no-RK 35NFE FID-50k over 3 Seed Ranges | 5-8のPNDM no-RK（固定係数AB）を35 steps／35 NFEに増やす。FID-50k（3 seed平均）は1.89947で追加samplerの中で最良だが、EDM Heun（1.83906）には届かない。 | ✓ |
| [5-10](5/10_spndm_35nfe_fid50k_3seeds/README.md) | Experiment 5-10: S-PNDM Sampler 35NFE FID-50k over 3 Seed Ranges | 5-2のS-PNDMをsteps=34にして35 NFEで評価する。FID-50k（3 seed平均）は2.21971で、19 NFEの3.14766から改善した。 | ✓ |
| [5-11](5/11_pndm_no_rk_18nfe_fid50k_3seeds/README.md) | Experiment 5-11: PNDM no-RK 18NFE FID-50k over 3 Seed Ranges | 固定係数AB（PNDM no-RK）を5-4の可変係数ABと同じ18 steps／18 NFEで評価し、AB係数の違いだけを比べる。FID-50k（3 seed平均）は1.99106で、可変係数（3.57748）より大幅に良い。 | ✓ |
| [5-12](5/12_ab_26nfe_fid50k_3seeds/README.md) | Experiment 5-12: Variable-Coefficient AB 26NFE FID-50k over 3 Seed Ranges | 可変係数ABを5-8の固定係数ABと同じ26 steps／26 NFEで評価し、AB係数の違いだけを比べる。FID-50k（3 seed平均）は2.45947で、固定係数（1.91000）より悪い。 | ✓ |
| [5-13](5/13_trajectory_distance_5-4_vs_5-11/README.md) | Experiment 5-13: Trajectory Distance for 5-4 vs 5-11 | 5-4（可変係数AB）と5-11（固定係数AB）の、高精度RK4参照軌道からの距離を測る（1024 seed）。最終出力の相対L2誤差は5-11が0.02286、5-4が0.07195で、固定係数の方が参照解に近い。 | ✓ |

資料（実験READMEなし）：[SAMPLER_EXPERIMENT_TEMPLATE.md](5/SAMPLER_EXPERIMENT_TEMPLATE.md)、`5/report/`

## 6（[グループREADME](6/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [6-1](6/1_edm_forward_norm_density/README.md) | Experiment 6-1: EDM Forward-Noising Norm Density | PNDM論文 Figure 2 のように、前向きノイズ付加後のデータのノルム分布を可視化する。 | ✓ |

## 7（[グループREADME](7/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [7-1](7/1_cifar10_cond_linear_fm_baseline/README.md) | Experiment 7-1: CIFAR-10 Class-Conditional Linear Flow Matching Baseline | EDMと同じDDPM++・学習設定でlinear-path flow matching（FM）を実装し、50 kimgのsmokeで確認する。lossの低下、checkpoint再読込、生成の決定性を確認した。pilotは7-2で実施し、full（200k）は未実施。 | ✓ |
| [7-2](7/2_cifar10_cond_linear_fm_pilot/README.md) | Experiment 7-2: CIFAR-10 Conditional Linear FM Pilot | 7-1のFMを20k kimgまで学習し、段階ごとにFID-5k（Euler 18 NFE）を測る。FIDは1k 42.62、3k 24.93、10k 13.27、20k 11.79と改善し、学習は正常に進んだ。 | ✓ |
| [7-3](7/3_fm_solver_nfe_and_edm_comparison/README.md) | Experiment 7-3: FM Solver/NFE Sweep and EDM Comparison | 20k学習のFMとEDMで、solverとNFEを変えてFID-5kを比べる。FMはNFEとともに改善し、Heunは5・9 NFEでわずかに良く、19・35 NFEでは逆。18 NFEのEulerではFM 11.79がEDM 12.28より良い。 | ✓ |
| [7-4](7/4_fm_edm_fid50k_confirmation/README.md) | Experiment 7-4: FM/EDM FID-50k Confirmation | 7-3の主要条件をFID-50k（seed 0-49999）で確認する。FM Eulerは18 NFEの7.286から35 NFEで5.302に改善したが、35 NFEのEDM Heun（2.990）との差は残った。 | ✓ |
| [7-5](7/5_fm_edm_trajectory_error/README.md) | Experiment 7-5: FM/EDM Trajectory and Integration Error | FMとEDMの各solverについて、高精度RK4参照解からの最終誤差（相対L2、256 seed）と経路形状を測る。35 NFEではFM Euler 0.123、FM Heun 0.054、EDM Euler 0.105、EDM Heun 0.033だった。 | ✓ |

## 8（[グループREADME](8/README.md)）

| ID | タイトル | 概要 | results |
|---|---|---|:-:|
| [8-1](8/1_restart_sampling_checkpoint_sweep/README.md) | Experiment 8-1: Restart Sampling Checkpoint Sweep | 学習途中の4 checkpointでRestart（K=1,2,5、t_min=0.06）と通常ODEをFID-1kで比べる予備評価。学習不足ほど改善が大きく（5.5kのK=5で同NFE比+2.21）、200kのK=5では悪化した。 | ✓ |
| [8-2](8/2_restart_k5_fid5k_confirmation/README.md) | Experiment 8-2: Selected Restart K FID-5k Confirmation | 8-1の傾向をFID-5kで再確認する。K=5の同NFE ODE比の改善は5.5k +1.58、20k +0.33で、120k・200kでは−0.11、−0.24と悪化した。 | ✓ |
| [8-3](8/3_restart_k5_fid50k_3seeds/README.md) | Experiment 8-3: Restart K=5 FID-50k over 3 Seed Ranges | 固定K=5（t_min=0.06、NFE 55）をFID-50k×3生成seed範囲で正式評価する。同NFE ODE比の改善は5.5k +1.718、20k +0.318、120k −0.048、200k −0.181で、学習後期では悪化する。 | ✓ |
| [8-4](8/4_restart_k_sweep_10k_interval/README.md) | Experiment 8-4: Restart K Sweep at 10k Checkpoint Intervals | 0-1の10k刻みcheckpointでK=0〜8（main 18、t_min=0.06）を走査し、最適Kの変化を調べる。FID-5kの最適Kはばらつきが大きく、FID-25k（80k以降）では多くが1。FID-50kは20k・40k・60kの部分評価で、走査は未完了。 | ✓ |
| [8-6](8/6_exact_10k_restart_k5_fid50k_1seed/README.md) | Experiment 8-6: Progressive Exact-10k Restart K=5 FID Evaluation | 0-1の10k〜120kでK=5（t_min=0.06）と同NFE ODEをFID-5k→25k→50kと段階的に比べる。FID-25kの改善は10k +0.712から100k +0.012へほぼ単調に縮小した（25kは10k〜100kのみ、50kは未実施）。 | ✓ |
| [8-7](8/7_trained_restart_k1_fid50k_1seed/README.md) | Experiment 8-7: Trained Restart K=1 vs. Equal-NFE ODE, FID-50k | 十分学習した2モデル（3-10の170k、公式pretrained）でRestart K=1と同NFE（39）のODEをFID-50kで比べる。差は−0.014、+0.006でほぼ同等だった。 | ✓ |
| [8-8](8/8_restart_tmin_comparison/README.md) | 実験8-8：Restart t_min 比較（K=5） | Restartのt_min=0.06と0.14をK=5で比べる（20k・40k・60k、3-10の170k）。FID-50kでは20k・40kで0.06、60k・170kで0.14がわずかに良い（差0.01〜0.09）。 | ✓ |
| [8-10](8/10_restart_fixed_nfe55_k_sweep/README.md) | 実験8-10：NFE=55固定で学習量とKを比較 | NFE=55固定・t_min=0.06で、K（K=5を除く）とmain stepsへの計算配分を比べる計画。本評価は未実行で、主設定はt_min=0.14の8-12に移った。 |  |
| [8-11](8/11_restart_fixed_main18_tmin014_k_sweep/README.md) | 実験8-11：N_main=18・t_min=0.14固定のK sweep | 主実験。0-1の20k〜200kでN_main=18・t_min=0.14に固定してK=0〜8を走査する。FID-50kの最適Kは20k:7、40k〜120k:3、140k以降は主に1で、学習が進むほど小さい。 | ✓ |
| [8-12](8/12_restart_fixed_nfe55_tmin014_k_sweep/README.md) | 実験8-12：NFE=55・t_min=0.14固定のK sweep | 補完実験。NFE=55固定（N_main=28−2K）・t_min=0.14でK=0〜8を走査する。FID-50kの最適Kは20k:8、40k〜140k:4〜7、160k以降:1。主スケジュールも変わるのでKだけの効果ではない。 | ✓ |
| [8-13](8/13_restart_optimal_k_fid50k_3seeds/README.md) | 実験8-13：最適KとK=0のFID-50k・3シード比較 | 8-11の最適KとK=0をFID-50k×3生成seed範囲で比べる。平均改善幅は20k 0.381、60k 0.105、180k 0.019、200k 0.013。100k・140kはこのPCではn=1（Slackでは完了と報告）。 | ✓ |
| [8-14](8/14_restart_subset30k_k_sweep/README.md) | 実験8-14：30,000枚学習の20k〜100kにおけるRestart K比較 | 0-2（30,000枚学習）の20k〜100kで8-11型・8-12型のK sweepを行い、全データ学習の結果と比べる。実行中（2026-11-03時点、100kのFID-50kは未完了）。 | ✓ |

資料（実験READMEなし）：`8/report/`

## reports

資料（実験READMEなし）：[2026-09-17_additional_experiments_report.md](reports/2026-09-17_additional_experiments_report.md)、[2026-10-02.plan.md](reports/2026-10-02.plan.md)、[slack_sources.md](reports/slack_sources.md)、`reports/assets/`、`reports/scripts/`

## 直下の資料

- [spring_semester_report.md](spring_semester_report.md)
