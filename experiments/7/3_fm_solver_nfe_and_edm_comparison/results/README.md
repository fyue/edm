# Runtime results

`run_fm_solver_nfe_comparison.sh`が`fm_solver_nfe_comparison.tsv`を生成する。
論文用の比較では、`comparison_group`が同じ行を同一NFEの組として扱う。

`scripts/plot_results.py`は結果TSVを読み、次の論文用図を生成する。

- `fid5k_solver_nfe_overview.png`: 確認・スライド用
- `fid5k_solver_nfe_overview.pdf`: 論文貼り込み用ベクター図
