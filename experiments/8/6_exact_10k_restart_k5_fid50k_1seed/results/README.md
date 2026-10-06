# Results

段階実行時に次を生成する。

- `restart_fid_summary.tsv`: condition × FID stageごとの生FIDと再現条件
- `restart_improvements.tsv`: stage・checkpoint内でpairedに計算したRestart改善幅
- `restart_k5_fid5000_overview.{png,svg}`: FID-5k結果
- `restart_k5_fid25000_overview.{png,svg}`: FID-25k結果
- `restart_k5_fid50000_overview.{png,svg}`: FID-50k結果

seed集合は`0-4999 ⊂ 0-24999 ⊂ 0-49999`。画像は共通poolへ追加し、前段階の画像を
再利用する。生成画像とlogはcommitしない。
