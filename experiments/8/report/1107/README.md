# 2026-11-07 研究進捗報告

RestartのN_main固定・NFE固定比較、3生成seed範囲の確認、200k以降の学習、30,000枚subsetの学習とK走査を報告する資料。`assets/` は保存済み結果のスナップショットであり、Slackの報告表の転記ではない。

## 保存内容と出典

- `fixed_main18__*`、`fixed_nfe55__*`：実験8-11・8-12の集計表と平均Kを含む図。FID-50k、生成seed範囲 `0-49999`。
- `seeds3__*`：実験8-13の `runs/early` と `runs/late` を、実験自身の `--aggregate-only` で統合した結果。
- `post200k_seeds3__*`：実験0-4の実験直下と `runs/pascal` を同様に統合した結果。
- `post200k__*`：実験0-3の通常ODEとRestart K=1の比較。`subset__fid_summary.tsv`：実験0-2の学習量とFID。
- `subset_main18__*`、`subset_nfe55__*`：実験8-14の各系列の結果表と、表から描いた図。スライド本文で比較するcheckpointは20k・40k・60k。
- `actual_tmin.json`：8-11・8-12の評価JSONに記録された実時刻とNFE。`progress.json`：保存時点の完了条件数。

3 seedは固定モデルからの3つの生成seed範囲（`0-49999`、`50000-99999`、`100000-149999`）であり、3回の独立学習ではない。複数rootを統合したこの報告用集計と、実験直下に残る以前の暫定集計は別の保存物。原データとの対応はスクリプトの集計元を確認する。議論の経緯は[Slack参照索引](../../../reports/slack_sources.md)を参照する。

## 再作成

リポジトリ直下で、PyTorch・NumPy・Matplotlib・Pillowなど実験の集計依存を含むedm環境を使用する。実験の評価JSONが必要で、GPU処理は行わない。

```bash
python experiments/8/report/1107/scripts/prepare_figures.py --help
python experiments/8/report/1107/scripts/prepare_figures.py
# 保存済みassetsを変えずに再作成を検証する場合
python experiments/8/report/1107/scripts/prepare_figures.py --outdir /tmp/edm-report1107
```

実験の `results/` は更新せず、報告用の出力先にだけ書き込む。subsetの図もコピーしたTSVから再生成するため、削除済みSlackフォルダやローカルSlack原本には依存しない。Kの平均は最小FIDとの差が0.02以下の候補の算術平均で、統計的同等性の検定ではない。

PowerPointの作成にはNode.js、pptxgenjs、jszip、および `scripts/apply_theme.js` を含む外部ディレクトリが必要。

```bash
NODE_PATH=<pptxgenjsとjszipを利用できるディレクトリ> \
PPTX_SKILL_DIR=<scripts/apply_theme.jsを含むディレクトリ> \
node experiments/8/report/1107/scripts/build_report.js
```

出力は本ディレクトリの `restart_progress_report_1107_claude.pptx`（Git対象外）。手書きの説明に対応する数値・完了数などは作成スクリプト内のassertで検証する。結果が変わった場合は、assertと本文・発表ノートを照合してから更新する。

## 検証状況

2026-11-05：Pythonのコンパイル・CLI表示とCPUでの図・集計の再作成を確認。現環境にNode.jsがないため、PowerPointのビルドとスライドの描画確認は未実施。
