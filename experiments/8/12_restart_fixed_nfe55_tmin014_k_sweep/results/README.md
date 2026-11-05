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

## 2パネル図：改善・悪化と良好なKの点分布

`--plot-kind near-optimal-dots`で横並び2パネルの図を出力する。左は既存の`FID(K) − FID(0)`ヒートマップ（青白赤、−0.4〜＋0.4共通）。右は横軸が学習量、縦軸がKで、同じcheckpoint・評価枚数の最小FIDとの差が0.01以内を濃青、0.01超〜0.02以内を薄青の点で示す。最良Kには大きな黒点を重ね、同値なら小さいKを選ぶ。点同士は線で結ばない。全Kが揃わないcheckpoint列は右を灰色にして点を表示しない。許容幅は表示上の基準で、統計的な同等性を意味しない。

FID-5k・25k・50kを各14×5.5インチ、PNGは180dpi（2520×990px）、SVGでも出力する。名前は既存の統合図名に`_near_optimal_dots`を付ける。既定動作・既存図・元データは保持し、GPU処理やFID再計算は行わない。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py \
  --plot-kind near-optimal-dots
```

### 良好なKの平均線

2パネル図の右には、各checkpoint・評価枚数で最良FIDとの差が0.02以内の全候補Kの算術平均をオレンジの折れ線と小さな丸印で重ねる。濃青・薄青の候補を等重みで1回ずつ数え、重ね描きした最良Kの黒点は二重計上しない。平均は丸めず表示し、全Kが揃わないcheckpointでは欠損として線を途切れさせる。平均Kは候補分布の中心であり、FIDを最小化するKとは異なる。再生成は上記の`--plot-kind near-optimal-dots`を使用する。

### 全Kを使うFID重み付き平均（別図）

`--plot-kind fid-weighted-k`は2パネル構成で、左の改善・悪化ヒートマップを維持し、右に最良Kの黒点と重み付き平均のオレンジ線を表示する。Best K以外の点は描かず、平均の計算には全K=0〜8を含める。重みは`w(K) = exp(-(FID(K) - min FID) / τ)`を合計1に正規化し、平均を`Σ K w(K)`で計算する。FID差による候補の除外は行わない（極端に小さい重みは浮動小数点で0になる場合がある）。全Kが揃わないcheckpointは灰色にし、平均線を途切れさせる。

`--weight-temperature`の既定値τ=0.02はFID単位の表示パラメータで、FID差の上限ではない。小さいほど最良Kを重視し、大きいほど全Kの等重み平均4に近づく。同値最良Kが複数あるとき、小さいτでの平均は同値候補の中心に近づく。重み付き平均Kは候補分布の中心であり、実際の最良Kや統計的推定値とは異なる。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py \
  --plot-kind fid-weighted-k --weight-temperature 0.02
```

出力はFID-5k・25k・50kのPNG/SVGで、従来の統合図名に`_fid_weighted_k_tau0p02`を付ける。温度を変更するとファイル名にも反映する。既存の図と元データは保持する。

### 4パネル図の候補点・Best K・単純平均K

`--plot-kind mean-k-overview`は従来の16×10インチの4パネル図を維持し、左下だけを`near-optimal-dots`の右パネルと同じ表示へ置き換える。濃青（FID差0.01以内）・薄青（0.01超〜0.02以内）の候補点、Best Kの黒点、単純平均Kのオレンジ線をすべて描く。タイトル・軸・凡例・グリッドも同じ描画関数を共用する。同じcheckpoint・評価枚数で最良FIDとの差が0.02以内のKを等重みで1回ずつ平均し、丸めず表示する。Best Kは同値なら小さいKを選ぶ。全Kが揃わないcheckpointでは線を途切れさせる。平均Kは近い性能を持つ候補の中心を示す記述量で、最適Kや統計的同等性の推定値ではない。

FID-5k・25k・50kのPNG（180dpi）/SVGを、元の統合図名に`_mean_k`を付けて出力する。元の図・2パネル図・結果表は保持し、GPU処理・FID再計算は行わない。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py \
  --plot-kind mean-k-overview
```
