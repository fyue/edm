# Experiment 8-4 Results

`scripts/plot_comparison.py`で8-11・8-12形式の比較図を再生成できる。

- `fixed_main18_tmin006_fid{5000,25000,50000}.{png,svg}`：20k～200kの10 checkpointsを対象に、
  K–FID曲線、K=0との差、最適Kを横並びにした図。±0.4の固定差分色を使用し、未完了は欠測表示。
- `comparison_fid_summary.tsv`：8-4を優先し、8-6・8-8・8-11から不足を補った採用値と出典。
  `--fid-stage`で指定したステージを保存する。元の生結果・集計表は書き換えない。

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
