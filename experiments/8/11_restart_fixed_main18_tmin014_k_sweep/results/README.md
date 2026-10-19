# 実験8-11：統合結果

2台分の全270評価を統合済み。対象は実験0-1の20k〜200k（20k刻み）、K=0〜8、FID-5k/25k/50k。各ステージ90件、最適Kの集計は全30組。

## 図と表

- `fixed_main18_tmin014_fid5000.{png,svg}`：FID-5k。
- `fixed_main18_tmin014_fid25000.{png,svg}`：FID-25k。
- `fixed_main18_tmin014_fid50000.{png,svg}`：FID-50k。
- 各図の左はcheckpoint別FID–K曲線、中央はFID(K)−FID(0)、右は学習量ごとの最適K。
- ヒートマップは3ステージ共通の−0.4〜＋0.4。中央の`0.0`は白、負（改善）は青、正（悪化）は赤。
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

# 統合後、0＝白・中央の共通色範囲で図を再生成する
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/plot_results.py
```

別PCでは`PYTHON`をそのPCのedm環境に変更する。統合結果を再生成するときは両方の`--result-roots`を指定する。実験ルートでの引数なし`--aggregate-only`は、そこにある生JSONの81件だけを集計する。

図だけの再生成は最後の`plot_results.py`コマンドで行う。`--summary`で統合JSON、`--outdir`で図の保存先を変更できる。入力は全270件が必要で、差分が±0.4を超えた場合はエラーにし、色範囲外の値を隠さない。実験ランナーの集計処理は従来の配色で描画するため、統合を再実行した後は描画専用スクリプトも実行する。ランナーと結果表・FID値・protocolは変更しない。

今回の統合では、全270組の網羅性・重複なし・protocolの整合性を確認し、各ステージ90件、最適K表30行、PNG/SVG各3ファイルの生成とPNGの読み取りを確認した。FID-50k図は目視でも確認済み。画像生成・FIDの再計算は実行していない。
