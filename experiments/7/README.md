# 実験7：Linear Flow MatchingとEDMの比較

## 問い

EDMと同じDDPM++構造・学習設定で、linear-path flow matching（FM）を学習する。FIDの差が、学習誤差・数値積分誤差・学習された経路のどれから来るのかを調べる。

## 主要な結論

出典：[reports/2026-09-17_additional_experiments_report.md](../reports/2026-09-17_additional_experiments_report.md)。

- FMは20k kimgまで正常に学習し、FID-5k（Euler 18 NFE）は1k 42.6155 → 20k 11.7926と改善した（7-2）。
- FID-50k（20k学習）：FM Euler 18 NFE 7.28617、FM Euler 35 NFE 5.30242、**EDM Heun 35 NFE 2.98971**。NFEを増やすとFMは改善するが、同じNFEではEDMとの差が残る（7-4）。
- 35 NFEでの参照解に対する最終相対L2誤差：FM Euler 0.12265、FM Heun 0.05377、EDM Euler 0.10535、EDM Heun 0.03333。FMでは、積分誤差の小さいHeunがFID-5kでEulerを上回らなかった。したがってFIDの差は積分誤差だけでは説明できず、学習誤差や学習された経路も関わると考えられる（7-3、7-5）。

## 実験一覧と役割

| ID | 役割 |
|---|---|
| 7-1 | FM baselineの実装と動作確認 |
| 7-2 | 20,000 kimgまでのpilot学習 |
| 7-3 | FMのsolver（Euler/Heun）×NFEとEDMの比較（FID-5k） |
| 7-4 | 7-3の主要傾向をFID-50kで確認 |
| 7-5 | FM/EDMの軌道誤差と積分誤差の分解（[results/README.md](5_fm_edm_trajectory_error/results/README.md)） |

## 未解決・次の候補

- 7-5のfinal結果の解釈は報告書にまとめられていない（要確認）。
