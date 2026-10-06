# Experiment 8-4 Results

実行後に次を生成する。

- `restart_fid_summary.tsv`: runnerが記録する条件ごとの生FIDと再現条件
- `restart_k_sweep.tsv`: checkpoint × KのFID、K=0との差、最良Kフラグ
- `optimal_k.tsv`: K=0～8が完走したcheckpointごとのFID最小K
- `restart_k_sweep_overview.png`: K走査と最適K推移の確認用グラフ
- `restart_k_sweep_overview.svg`: 編集可能なvector版グラフ
- `fid_by_checkpoint_and_k.tsv`: checkpointごとのK=0～8を横持ちにした分析用表
- `fid_by_checkpoint_and_k.md`: 最小FIDを強調した閲覧用表
- `fid_vs_k_by_checkpoint.{png,svg}`: checkpoint別のFID–K曲線
- `fid_delta_vs_k0_heatmap.{png,svg}`: K=0との差を示すヒートマップ

FID-5kは一次走査値であり、最終的な主張には選択したKと近傍を独立seedの
FID-50kで確認する。

`fid25k/`には、指定したcheckpointの既存FID-5k画像を再利用して追加20,000枚を生成する
FID-25k拡張のmanifest、生FID、集計表、グラフを保存する。
