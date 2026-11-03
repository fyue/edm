# 加藤先生とのRestart研究の議論・必要図

2026-11-03 JST時点の整理。まず[概要と合意事項](summary.md)、次に[必要図一覧](figures.md)と[対応事項](actions.md)を参照する。

指定スレッドは2026-10-17の親メッセージから2026-11-02の最終返信まで全12メッセージを取得した。指定URLが指す11月1日の返信だけでなく、前後の指示と更新も確認している。DMは10月1日以降の履歴を取得し、背景として9月11日以降のRestart関連発言も確認した。10月分の取得範囲内で最新のDM本文は10月20日。9月以前のDM全履歴を網羅した要約ではない。

## ファイル

- [summary.md](summary.md)：研究の目的、最新方針、重要な結果、解釈上の区別。
- [figures.md](figures.md)：図8組の用途、出典、計算式、埋め込みプレビュー。
- [actions.md](actions.md)：報告済み／ローカル確認済み／要確認を区別した作業リスト。
- `sources/`：Slackコネクターから取得した本文と添付メタデータ。各発言に直接リンクを追加した。Slack添付バイナリそのものはダウンロードしていない。
- `data/`：ローカル集計TSVのスナップショット、Slack報告値の転記、図の計算表。
- `figures/`：PNGとSVG。01〜05は保存済み実測TSVから再描画、06〜07はSlackに報告された集計値から描画、08は既存の実験8-8の図をコピー。
- [scripts/build_figures.py](scripts/build_figures.py)：取得済みTSVからのCPU専用再描画。既存の実験結果・コードを変更しない。

## 再描画

edm環境で実行する。新規画像生成・FID計算・GPU処理は行わない。

```bash
python experiments/slack/scripts/build_figures.py --help
MPLCONFIGDIR=/tmp/edm-slack-matplotlib python experiments/slack/scripts/build_figures.py
```

01〜07と図の計算表を再生成する。08は元のコピーを保持する。`sources/local_artifact_manifest.json`にコピー元とSHA256、`sources/generated_artifact_manifest.json`に最終成果物のSHA256を記録した。Slackの3シード報告値の出典は`figures.md`を参照する。

## 保存先と取得上の区別

依頼された`/home/kyoukinkin/projects/edm`はこのPCに存在しないため、現在のリポジトリである`/data01/kyoukinkin/projects/edm/experiments/slack`に保存した。実体は`/data00/users/kyoukinkin/projects/edm/experiments/slack`。別リポジトリやホーム以下に代替ツリーは作成していない。

Slackの報告内容と、このPCへ同期されたデータの完了状況は一致しない箇所がある。特に3シード評価は、報告済みの数値から図を作ったことを明記しており、未同期のseed別結果を復元・推定してはいない。図はSlack添付画像の完全な複製ではない。
