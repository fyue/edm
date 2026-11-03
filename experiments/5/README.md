# 実験5：Sampler比較（PNDM・S-PNDM・RK4・Adams-Bashforth）

## 問い

公式pretrainedのEDM（CIFAR-10 class-conditional）に、PNDM系・RK4・ABのsamplerをEDM座標系へ移植して、NFEとFIDの関係を比較する。可変係数ABと固定係数AB（PNDM no-RK）の差はどこから来るか。

## 主要な結論

すべて`S_churn=0`、FID-50kの3 seed範囲平均。出典：[report/2026-09-01_sampler_comparison_report.md](report/2026-09-01_sampler_comparison_report.md)。

| 実験 | sampler | NFE | FID平均 |
|---|---|---:|---:|
| 基準 | EDM Heun | 35 | **1.83906** |
| 5-9 | PNDM no-RK（固定係数AB） | 35 | 1.89947（追加sampler中で最良） |
| 5-8 | PNDM no-RK | 26 | 1.91000 |
| 5-5 | PNDM（RK warmup） | 35 | 1.91047 |
| 5-3 | RK4 | 69 | 1.92574 |
| 5-11 | PNDM no-RK | 18 | 1.99106 |
| 5-1 | PNDM | 27 | 1.99152 |
| 5-7 / 5-12 / 5-4 | 可変係数AB | 35 / 26 / 18 | 2.12443 / 2.45947 / 3.57748 |
| 5-10 / 5-2 | S-PNDM | 35 / 19 | 2.21971 / 3.14766 |
| 5-6 | PNDM-linear（linear sigma schedule） | 27 | 98.1024（失敗） |

- 同じNFEでは、固定係数AB（等間隔用の係数）が、EDM rho scheduleに合わせた可変係数ABより良い（18 NFE：5-11 1.99106 vs 5-4 3.57748、26 NFE：5-8 1.91000 vs 5-12 2.45947）。
- PNDMのRK4 warmupは不要だった。warmupを除いた5-8（26 NFE）は、warmupありの5-5（35 NFE）と同等（1.91000 vs 1.91047）。
- 追加samplerはどれもEDM Heun（35 NFE、1.83906）に届かない。RK4はNFEを69に増やしても1.92574。
- 5-6の失敗は、linear scheduleで低ノイズ側のstepが足りない（最後の1 stepでσ=4.44から0へ飛ぶ）ためと考えている（[report/2026-08-28.md](report/2026-08-28.md)）。
- seed範囲間のstdはどの条件も約0.006〜0.018（5-6のみ0.19）。
- 5-13：固定係数ABは高精度参照軌道にも近い（最終相対L2誤差 0.02286、可変係数ABは0.07195）。「参照軌道から外れるから画質が良い」という仮説とは合わない。出典：[reports/2026-09-17_additional_experiments_report.md](../reports/2026-09-17_additional_experiments_report.md)。
- 係数の比較は[report/2026-09-01_ab_coefficients_5-4_vs_5-11.md](report/2026-09-01_ab_coefficients_5-4_vs_5-11.md)にある。

## 実験一覧と役割

すべて公式pretrained（`edm-cifar10-32x32-cond-vp`）で、学習は行わない。各実験READMEのH1直後に目的と結論がある。

| 比較の軸 | 実験 |
|---|---|
| sampler単体の評価（18 steps） | [5-1](1_pndm_fid50k_3seeds/README.md) PNDM、[5-2](2_spndm_fid50k_3seeds/README.md) S-PNDM、[5-3](3_rk4_fid50k_3seeds/README.md) RK4、[5-4](4_ab_fid50k_3seeds/README.md) 可変係数AB |
| 35 NFEに揃えた比較 | [5-5](5_pndm_35nfe_fid50k_3seeds/README.md) PNDM、[5-7](7_ab_35nfe_fid50k_3seeds/README.md) 可変係数AB、[5-9](9_pndm_no_rk_35nfe_fid50k_3seeds/README.md) PNDM no-RK、[5-10](10_spndm_35nfe_fid50k_3seeds/README.md) S-PNDM |
| RK4 warmupの有無 | [5-8](8_pndm_no_rk_warmup_fid50k_3seeds/README.md)（5-5からwarmupを除く） |
| AB係数だけの比較（同steps・同NFE） | 18 NFE：[5-4](4_ab_fid50k_3seeds/README.md) vs [5-11](11_pndm_no_rk_18nfe_fid50k_3seeds/README.md)、26 NFE：[5-12](12_ab_26nfe_fid50k_3seeds/README.md) vs [5-8](8_pndm_no_rk_warmup_fid50k_3seeds/README.md) |
| sigma schedule | [5-6](6_pndm_linear_fid50k_3seeds/README.md)（linear schedule、失敗） |
| 軌道誤差の解析 | [5-13](13_trajectory_distance_5-4_vs_5-11/README.md)（5-4と5-11の参照軌道からの距離） |

新しいsampler実験を追加するときの雛形は[SAMPLER_EXPERIMENT_TEMPLATE.md](SAMPLER_EXPERIMENT_TEMPLATE.md)。

各実験の `scripts/run_*.sh` は、直接実行すると `CUDA_VISIBLE_DEVICES` 未設定時にGPU 0を使う（現在のGPU設定ルール以前の実装）。再実行するときは `scripts/schedule_after_gpu_free.sh`、それがない5-1〜5-4は共通の `scripts/wait_for_gpu_and_run.sh` に `RUN_SCRIPT`・`EXP_ROOT` を渡して起動し、ローカルのGPU設定を使う。

## 未解決・次の候補

- 固定係数ABの方が積分精度が高い理由（係数の外挿特性とEDMの低ノイズ域の関係）は未解明。
