# Results

`restart_fid_summary.tsv`は実行時に作成される。各行にはcheckpoint、通常Heunまたは
Restart、main steps、Restart設定、総NFE、seed範囲、FID、所要時間、入出力pathを
記録する。

`restart_improvements.tsv`はRestartのFID改善幅、`restart_fid_curves.{png,svg}`は
checkpoint別のFID対NFE、`restart_improvements.{png,svg}`は学習進行に対する改善幅を
示す。これらは`../scripts/summarize_results.py`と`../scripts/plot_results.py`で再生成する。

大容量の生成画像とログはコミットしない。
