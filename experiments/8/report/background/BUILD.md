# 背景説明資料のビルド

日付に依存しない背景説明資料。元の `1024/background/` から図と作成スクリプトを移動し、スライド中の説明も更新した。

リポジトリ直下で実行する。保存済みの `assets/` を使えば論文の再取得は不要。

```bash
NODE_PATH=<pptxgenjsとjszipを利用できるディレクトリ> \
PPTX_SKILL_DIR=<scripts/apply_theme.jsを含むディレクトリ> \
node experiments/8/report/background/scripts/build_report.js
```

出力は `experiments/8/report/restart_background_claude.pptx`（Git対象外）。Node.js、pptxgenjs、jszip、外部の `apply_theme.js` が必要。

図を取り直す場合は、Pillow、curl、pdftoppmがある環境で次を実行する。論文PDFのキャッシュは `/tmp/deck_bg/papers`、抽出位置と論文IDはスクリプト内にある。キャッシュがなければネットワーク接続が必要。

```bash
python experiments/8/report/background/scripts/prepare_figures.py
```
