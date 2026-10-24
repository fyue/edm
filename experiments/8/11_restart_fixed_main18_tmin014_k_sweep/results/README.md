# 実験8-11：統合結果

2台分の全270評価を統合済み。対象は実験0-1の20k〜200k（20k刻み）、K=0〜8、FID-5k/25k/50k。各ステージ90件、最適Kの集計は全30組。

## 図と表

- `fixed_main18_tmin014_fid5000.{png,svg}`：FID-5k。
- `fixed_main18_tmin014_fid25000.{png,svg}`：FID-25k。
- `fixed_main18_tmin014_fid50000.{png,svg}`：FID-50k。
- 各統合図は2行×2列。左上はcheckpoint別FID–K曲線、右上はFID(K)−FID(0)のヒートマップ、左下は学習量ごとの最適K（20k〜200kの20k刻み）、右下はFID(K=0)とK別改善幅の散布図。16×10インチ、PNGは180dpiで、ステージ・固定条件は共通タイトルに表示する。
- ヒートマップは3ステージ共通の−0.4〜＋0.4。中央の`0.0`は白、負（改善）は青、正（悪化）は赤。
- `fid_baseline_vs_improvement_fid{5000,25000,50000}.{png,svg}`：横軸FID(K=0)、縦軸FID(K=0)−FID(K)の散布図。正が改善、負が悪化。K=1〜8を共通の色で表示し、各Kの10checkpoint、計80点を線で結ばず描く。縦軸は3ステージの全改善幅と0を含む範囲に上下5%の余白を付けて統一する。横軸はステージごとに調整する。
- `restart_fid_summary.tsv/.json`：全270件の完了結果。詳細protocolはJSONに保持。
- `fid_by_checkpoint_and_k.tsv`：FIDとK=0との差。
- `optimal_k.tsv`：checkpoint・ステージ別の最適K。同値なら小さいK。

## 再生成

生の完了JSONは2か所に分かれている。実験ルートの`results/`が160k〜200kの81件、`runs/20k-140k/results/`が20k〜140kの189件。以下をリポジトリ直下で実行する。

```bash
PYTHON=/home/kyoukinkin/data/conda-envs/edm/bin/python \
experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh \
  --aggregate-only --result-roots \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/runs/20k-140k

# 統合後、散布図を含む2行×2列の統合図を再生成する
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/plot_results.py

# FID(K=0)とK別改善幅の散布図だけを追加・再生成する
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/plot_results.py \
  --plot-kind baseline-improvement
```

別PCでは`PYTHON`をそのPCのedm環境に変更する。統合結果を再生成するときは両方の`--result-roots`を指定する。実験ルートでの引数なし`--aggregate-only`は、そこにある生JSONの81件だけを集計する。

図だけの再生成には`plot_results.py`を使う。`--summary`で統合JSON、`--outdir`で図の保存先を変更できる。入力は全270件が必要で、差分が±0.4を超えた場合はエラーにし、色範囲外の値を隠さない。実験ランナーの集計処理は従来の配色で描画するため、統合を再実行した後は描画専用スクリプトも実行する。ランナーと結果表・FID値・protocolは変更しない。

`--plot-kind`は既定の`overview`で2行×2列の4パネル統合図、`baseline-improvement`で単独散布図、`all`で両方を出力する。統合図と単独散布図は同じ描画関数を使い、点・配色・凡例・軸範囲を統一する。散布図の色はKの区別を表し、ヒートマップの改善・悪化を表す色とは意味が異なる。

今回の統合では、全270組の網羅性・重複なし・protocolの整合性を確認し、各ステージ90件、最適K表30行、PNG/SVG各3ファイルの生成とPNGの読み取りを確認した。FID-50k図は目視でも確認済み。画像生成・FIDの再計算は実行していない。

## 最良FIDに近いKの分布（試作）

`--plot-kind near-optimal`で、左下を最良FIDとの差のヒートマップに置き換えた2×2図を追加出力する。横軸は学習量、縦軸はK=0〜8。各checkpoint・評価枚数で`FID(K) − min FID`を計算し、差0の濃青から0.05の白まで共通スケールで表示する。0.05超は白で、存在するときはカラーバー上端に範囲超過マークを付ける。黒点は最良K（同値なら小さいK）で、線では結ばない。K=0〜8が揃わない列は灰色とし、黒点を表示しない。

色範囲0〜0.05は近接する候補を見分けるための表示設定で、統計的な同等性を示すものではない。既定図と元のFID・集計表は維持する。出力名は既存の統合図名に`_near_optimal`を付けたPNG/SVGで、FID-5k・25k・50kの計6ファイル。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/plot_results.py \
  --plot-kind near-optimal
```

GPU処理やFID再計算は不要。
