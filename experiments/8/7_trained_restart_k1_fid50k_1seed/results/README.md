# Results

実行時に生成する。

- `restart_fid_summary.tsv`: 4条件の単一seed範囲FIDと再現条件、生成・FID所要時間。
- `restart_improvements.tsv`: checkpoint別の`FID(ODE) - FID(Restart)`。正値が改善。
- `restart_k1_fid50k_overview.{png,svg}`: FIDと改善幅。完了した比較だけを表示する。

seed範囲は`0-49999`の1組。seed間平均・標準偏差は算出しない。
実測値は実験実行後に記録する。生成画像・モデルcache・logはcommitしない。
