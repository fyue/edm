# 実験7：Linear Flow MatchingとEDMの比較

## 問い

EDMと同じDDPM++構造・学習設定で、linear-path flow matching（FM）を学習する。FIDの差が、学習誤差・数値積分誤差・学習された経路のどれから来るのかを調べる。

## 主要な結論

出典：[reports/2026-09-17_additional_experiments_report.md](../reports/2026-09-17_additional_experiments_report.md)。

- FMは20k kimgまで正常に学習し、FID-5k（Euler 18 NFE）は1k 42.6155 → 20k 11.7926と改善した（7-2）。
- FID-50k（20k学習）：FM Euler 18 NFE 7.28617、FM Euler 35 NFE 5.30242、**EDM Heun 35 NFE 2.98971**。NFEを増やすとFMは改善するが、同じNFEではEDMとの差が残る（7-4）。
- 同じsolver・NFEなら差は小さい。Euler 18 NFEのFID-5kはFM 11.7926、EDM 12.2757で、FMの方がやや良い。EDMの優位は、35 NFEでEDM Heunを使ったとき（FID-5k 7.6149）に現れる（7-3）。
- FM内のsolver比較（FID-5k）：Heunは5・9 NFEでEulerよりわずかに良い（25.44 vs 26.59、15.40 vs 16.20）が、19・35 NFEでは逆（11.67 vs 11.57、10.16 vs 9.86）。7-3の読み方では0.1程度の差は有意としない（7-3）。
- 35 NFEでの参照解に対する最終相対L2誤差：FM Euler 0.12265、FM Heun 0.05377、EDM Euler 0.10535、EDM Heun 0.03333。FMでは積分誤差の小さいHeunでもFIDは良くならない。したがってFIDの差は積分誤差だけでは説明できず、学習誤差や学習された経路も関わると考えられる（7-3、7-5）。
- 低NFEでは、EDM Heunの誤差が大きい（5 NFEで4.55、9 NFEで0.571。FM Heunは0.539、0.346）。RK4参照解のcoarse/fine差はFM 6.5e-4、EDM 2.5e-6で、どの比較誤差よりも十分小さい（[7-5 final](5_fm_edm_trajectory_error/results/final/README.md)）。
- EDM側は、基準学習（[1-cifar10-...](../1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/README.md)）の20k checkpointを使っている。FMは7-2の20k。

## 実験一覧と役割

各実験READMEのH1直後に目的と結論がある。

| ID | 役割 |
|---|---|
| [7-1](1_cifar10_cond_linear_fm_baseline/README.md) | FM baselineの実装とsmoke（50 kimg）。`PROFILE=full`（200k）は未実施 |
| [7-2](2_cifar10_cond_linear_fm_pilot/README.md) | 20,000 kimgまでのpilot学習（FID-5k） |
| [7-3](3_fm_solver_nfe_and_edm_comparison/README.md) | FMのsolver（Euler/Heun）×NFEとEDMの比較（FID-5k） |
| [7-4](4_fm_edm_fid50k_confirmation/README.md) | 7-3の主要傾向をFID-50kで確認（1 seed範囲） |
| [7-5](5_fm_edm_trajectory_error/README.md) | FM/EDMの数値積分誤差と経路形状（[final結果](5_fm_edm_trajectory_error/results/final/README.md)） |

## 未解決・次の候補

- 報告書（2026-09-17）は7-5の35 NFEの最終誤差だけを扱っている。経路形状（straightness、chordからのずれ）と低NFEでの結果の解釈は未整理。経路形状の値は状態空間と時間パラメータに依存するため、FMとEDMの直接比較には注意が要る。
- FMの200k学習（7-1の`PROFILE=full`）を行うかは未定。
