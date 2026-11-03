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

- 同じNFEでは、固定係数AB（等間隔用の係数）が、EDM rho scheduleに合わせた可変係数ABより良い。
- 5-13：固定係数ABは高精度参照軌道にも近い（最終相対L2誤差 0.02286、可変係数ABは0.07195）。「参照軌道から外れるから画質が良い」という仮説とは合わない。出典：[reports/2026-09-17_additional_experiments_report.md](../reports/2026-09-17_additional_experiments_report.md)。
- 係数の比較は[report/2026-09-01_ab_coefficients_5-4_vs_5-11.md](report/2026-09-01_ab_coefficients_5-4_vs_5-11.md)にある。

## 実験一覧と役割

上の表に加え、5-13が5-4と5-11の軌道距離を測る解析。新しいsampler実験を追加するときの雛形は[SAMPLER_EXPERIMENT_TEMPLATE.md](SAMPLER_EXPERIMENT_TEMPLATE.md)。

## 未解決・次の候補

- 固定係数ABの方が積分精度が高い理由（係数の外挿特性とEDMの低ノイズ域の関係）は未解明。
