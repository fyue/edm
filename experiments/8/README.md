# 実験8：Restart Sampling

## 問い

学習不足のEDMモデルに対し、Restart Samplingでサンプリング品質を改善できるか。学習量（checkpoint）と最適なRestart回数Kの関係はどうなるか。学習データが少ない場合（実験0-2の30,000枚subset）も同じように改善できるか。後者2つは別の問題として扱う（[slack/summary.md](../slack/summary.md)）。

## 主要な結論

- **固定K=5（t_min=0.06、NFE=55）**：3 seed平均で、5.5kと20kでは同NFEの通常ODEより改善（+1.7179、+0.3176）、120kと200kでは悪化（−0.0483、−0.1808）。出典：8-3、[reports/2026-09-17_additional_experiments_report.md](../reports/2026-09-17_additional_experiments_report.md)。
- **t_min**：初期checkpointでは0.06、170kでは0.14が有利。2026-10-07以降の主設定は **t_min=0.14、t_max=0.30**。0.06は付録的な比較として残す（8-8、[slack/summary.md](../slack/summary.md)）。
- **主実験 8-11（N_main=18固定、NFE=35+4K）**：FID-50k（seed 0-49999）での最適Kは 20k:7、40k〜120k:3、140k:1、160k:3、180k・200k:1。学習が進むほど最適Kが小さい傾向。出典：[11_.../results/optimal_k.tsv](11_restart_fixed_main18_tmin014_k_sweep/results/optimal_k.tsv)。
- **補完実験 8-12（NFE=55固定、N_main=28−2K）**：最適Kは 20k:8、40k:6、60k:4、80k:6、100k:4、120k・140k:7、160k〜200k:1。Kだけの効果ではなく主スケジュールも変わる点に注意。出典：[12_.../results/optimal_k.tsv](12_restart_fixed_nfe55_tmin014_k_sweep/results/optimal_k.tsv)。
- **3 seed確認 8-13（8-11の最適K vs K=0）**：平均改善幅は 20k 0.38073、60k 0.10470、180k 0.01933、200k 0.01344。baseline FIDが悪いモデルほど改善が大きい。ローカルの [improvement_stats.tsv](13_restart_optimal_k_fid50k_3seeds/results/improvement_stats.tsv) では100k・140kがn=1（Slackでは6点すべて完了と報告。同期の確認が必要）。
- **30,000枚学習 8-14**：学習不足段階では改善し、十分学習すると悪化する傾向。40kはイレギュラー。指導教員の解釈は「データ不足そのものはRestartでは救えない」。この解釈は今回の条件に限る（[slack/summary.md](../slack/summary.md)）。
- **Kの傾向図**：最良FIDとの差が0.02以下のKを単純平均する方式を優先する（2026-11-02）。

## 実験一覧と役割

| ID | 役割 |
|---|---|
| [8-11](11_restart_fixed_main18_tmin014_k_sweep/README.md) | **主実験**。N_main=18、t_min=0.14、K=0〜8、20k〜200k |
| [8-12](12_restart_fixed_nfe55_tmin014_k_sweep/README.md) | 補完。NFE=55固定、t_min=0.14 |
| [8-13](13_restart_optimal_k_fid50k_3seeds/README.md) | 8-11の最適KをFID-50k・3 seed範囲で確認 |
| [8-14](14_restart_subset30k_k_sweep/README.md) | データ不足ケース（実験0-2の30k subset）でのK sweep。進行中 |
| [8-10](10_restart_fixed_nfe55_k_sweep/README.md) | NFE=55固定、t_min=0.06（旧設定） |
| [8-8](8_restart_tmin_comparison/README.md) | t_min 0.06と0.14の比較（K=5） |
| [8-4](4_restart_k_sweep_10k_interval/README.md) | 10k刻みcheckpointでのK sweep（t_min=0.06） |
| 8-6, 8-7 | 0-1の10k刻み系列でのK=5、十分学習したモデルでのK=1と同NFE ODEの比較 |
| 8-1〜8-3 | 初期のスクリーニング（FID-1k、FID-5k）から固定K=5のFID-50k・3 seed評価まで |

8-5と8-9は欠番。報告資料は [report/](report/)（日付別の`1003`〜`1024`）と [../slack/](../slack/) にある。200k以降のRestart K=1は実験[0-3・0-4](../0/README.md)にある。

## 未解決・次の候補

- 8-13（100k・140k）と0-4のローカル集計がn=1。元PCのseed別完了JSONと照合する（[slack/actions.md](../slack/actions.md)）。
- 8-14は100kのFID-50kが両系列とも未完了。main18は100kが未着手、nfe55は80k・100kともFID-5kまで（2026-11-03時点のローカル状態）。
