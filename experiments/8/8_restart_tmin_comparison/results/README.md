# 実験8-8の結果

実測結果は `scripts/run.sh` によるFID評価後に保存する。

実装時の検証内容は [validation.md](validation.md) に記録する。拡張前の20k/40kのFID-5k/25k結果を保持し、60kとFID-50kを追加する。

- `restart_fid_summary.tsv` / `.json`: 条件・ステージ別の生結果、実行時間、画像再利用元、protocol ID。
- `tmin_comparison.tsv` / `.json`: チェックポイント・ステージ別の両FIDと差分。未完了の対は `NA`。
- `restart_tmin_fid{5000,25000,50000}.{png,svg}`: FID棒グラフと差分。
- `restart_tmin_fid{5000,25000,50000}_lines.{png,svg}`: t_min別に20k・40k・60kのFID点を結ぶ2本の折れ線図。
- `completion.json`: 本評価・完了チェックの状態。`status=complete` は全18件・画像集合・3種類の折れ線図・既存8件の保持を確認済みであることを示す。
- `extension_baseline.json`: 拡張前8件の結果JSONのSHA256。

`delta = FID(0.06) - FID(0.14)`。正なら0.14が良い。既存実験8-4のFID値は転記せず、画像を再利用した場合も再評価する。
