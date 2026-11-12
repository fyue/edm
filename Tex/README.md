# 論文原稿

upLaTeX（`jsarticle`）+ upBibTeX + dvipdfmx で日本語論文をビルドします。

## 構成

- `main.tex`: プリアンブルと章の読み込み（`jsreport`）
- `chapters/`: 章ごとの本文。`main.tex` から `\input` する
- `references.bib`: 参考文献（`junsrt` スタイル）
- `figures/`: 図。dvipdfmx は SVG を読めないので、PDF か PNG に変換して置く
- `latexmkrc`: ビルド設定。出力はすべて `build/`（git 管理外）
- `sample.tex`: 参考用の過去の宿題

## 後で修正が必要な箇所

未完了の実験に依存する記述などには `TODO(要修正)` を付けます。本文に見せるものは
`\todo{...}`（PDFに赤字で表示）、見せないものは `% TODO(要修正): ...` と書きます。

```bash
grep -rnE 'TODO\(要修正\)|\\todo\{' Tex/ --include='*.tex'
```

## 必要なもの

TeX Live 2026 を `/data01/kyoukinkin/texlive/2026` にインストールし、
`~/.bashrc` で `bin/x86_64-linux` を `PATH` に追加しておきます。

## ビルド

```bash
cd Tex
latexmk              # build/main.pdf を生成
latexmk sample.tex   # 別ファイルを指定してビルド
latexmk -pvc         # 保存のたびに再ビルド
latexmk -c           # 中間ファイルを削除（PDF は残す）
```

VSCode では LaTeX Workshop 拡張を使います。`.tex` を保存すると自動でビルドされ、
`build/` の PDF がエディタのタブに表示されます。
