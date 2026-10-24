# 実験8-12の結果

両PCの本評価が完了し、2026-10-20に全270件を統合した。FID-5k・25k・50kはそれぞれ90条件（20k〜200k、20k刻み×K=0〜8）。

## 統合元と出力

- `../runs/20k-120k/results/`：RTX 8000側、54条件×3ステージ＝162件。
- `../runs/140k-200k/results/`：Pascal側、36条件×3ステージ＝108件。
- `restart_fid_summary.tsv/.json`：全270件のFID・生成条件・再現情報。
- `fid_by_checkpoint_and_k.tsv`：FIDとK=0との差。負なら改善。
- `optimal_k.tsv`：全9候補が揃った30組の最適K。同値なら小さいK。
- `fixed_nfe55_tmin014_fid{5000,25000,50000}.{png,svg}`：全10 checkpointの統合結果図、計6ファイル。

図は2026-10-21に最新8-11と同じ16×10の2×2形式へ更新した。左上はFID–K曲線（上部5列の凡例）、右上は差分ヒートマップ、左下は学習量と最適K、右下はK=0のFIDとRestart改善量の散布図。差分ヒートマップは青→純白→赤、−0.4〜+0.4固定で、範囲外にカラーバーの三角マークを付ける。Kは0〜8の整数刻み、学習量は20k〜200kの20k刻み。

散布図は横軸`FID(K=0)`、縦軸`FID(K=0)−FID(K)`で、正なら改善。K=1〜8を色で区別し、改善量0の破線と薄いグリッドを付ける。全3ステージ共通の縦軸範囲は、全改善量と0を含めて上下に5％の余白を付ける。単独散布図は追加しない。結果データとランナー・protocolは変更しない。

## 再生成

リポジトリ直下でedm環境のPythonを使用する。

```bash
experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/run.sh \
  --aggregate-only --result-roots \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/140k-200k

python experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py
```

`--aggregate-only`の後に専用描画スクリプトを実行して図の表示を揃える。別Python環境を使う場合は前者に`PYTHON=/path/to/edm/bin/python`を設定し、後者も同じPythonで実行する。GPU処理・画像生成は不要。

## 統合時の検証

元結果のprotocol・モデル・参照統計・コードSHA256の整合性、欠落・重複なしを確認した。統合JSONの全270件が元結果と全フィールド一致すること、270件の差分と30組の最適Kが再計算値と一致することを検証した。元結果JSON・TSVおよび実験スクリプトは統合前後のSHA256が一致する。

## 最良FIDに近いKの分布（試作）

`--plot-kind near-optimal`で、左下を最良FIDとの差のヒートマップに置き換えた2×2図を追加出力する。横軸は学習量、縦軸はK=0〜8。各checkpoint・評価枚数で`FID(K) − min FID`を計算し、差0の濃青から0.05の白まで共通スケールで表示する。0.05超は白で、存在するときはカラーバー上端に範囲超過マークを付ける。黒点は最良K（同値なら小さいK）で、線では結ばない。K=0〜8が揃わない列は灰色とし、黒点を表示しない。

色範囲0〜0.05は近接する候補を見分けるための表示設定で、統計的な同等性を示すものではない。既定図と元のFID・集計表は維持する。出力名は既存の統合図名に`_near_optimal`を付けたPNG/SVGで、FID-5k・25k・50kの計6ファイル。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py \
  --plot-kind near-optimal
```

GPU処理やFID再計算は不要。
