# Results

実行時に次を生成する。

- `restart_fid_summary.tsv`: condition × seed範囲ごとの生FIDと再現条件
- `restart_improvements.tsv`: seed範囲内でpairedに計算したRestart改善幅
- `restart_k5_fid50k_3seed_stats.tsv`: 3 seed範囲の平均・標本標準偏差・最小・最大
- `restart_k5_fid50k_3seed_overview.png`: 発表・確認用グラフ
- `restart_k5_fid50k_3seed_overview.svg`: 編集可能なvector版グラフ

大容量の生成画像とlogはcommitしない。
