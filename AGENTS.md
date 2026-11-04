# Repository Guidelines

## プロジェクト構成

主要なCLIはリポジトリ直下にあります。`train.py` は学習、`generate.py` は画像生成、`fid.py` はFID評価、`dataset_tool.py` はデータセット変換を担当します。モデル、損失、拡張、データセット、学習ループは `training/`、共通機能は `torch_utils/` と `dnnlib/` に配置します。再利用する自動化は `scripts/`、実験固有の設定・スクリプト・結果・記録は `experiments/<グループ>/<実験名>/` にまとめてください。CLI資料と画像は `docs/` に置きます。論文原稿は `Tex/` に置き、ビルド方法は `Tex/README.md` を参照してください。生成画像、チェックポイント、ダウンロード済みモデル、大容量のFID一時データはコミットしないでください。

## 実験の把握

実験について調べるときは、ディレクトリを探索する前に `experiments/README.md`（グループの役割・系譜・最新方針の参照先）と `experiments/INDEX.md`（全実験一覧、自動生成）を読み、該当するグループの `README.md`、実験の `README.md`、`results/` の順に絞り込んでください。`training-runs/`、`fid-evals/`、`generated/`、`images/`、`logs/` は再帰的に読みません。

新しい実験を作ったときや結果が確定したときは、実験READMEのH1直後の段落に目的（確定後は結論も）を1〜2文で書き、グループREADMEの「主要な結論」と「実験一覧と役割」を出典付きで更新し、`python scripts/build_experiment_index.py` でINDEXを再生成してください。INDEXは手で編集しません。`--check` で更新漏れを確認できます。

## ビルド・テスト・開発コマンド

- `conda env create -f environment.yml -n edm && conda activate edm`: Python 3.8/3.9環境を作成します。
- `python example.py`: 学習済みモデルのデモを実行します。GPUとネットワーク接続が必要です。
- `python generate.py --help`、`python train.py --help`、`python fid.py --help`: CLIのインポートとオプションを確認します。
- `python train.py --outdir=/tmp/edm-dry-run --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --dry-run`: 学習を開始せず設定を検証します。
- `docker build -t edm:latest .`: NVIDIA PyTorch開発イメージを作成します。

## GPU実験の実行

GPUを使う学習、生成、FID評価などの実験スクリプトやプログラムは、原則としてGPUが空くまで待機してから開始する処理を備えてください。待機処理を実験ごとに複製せず、共通の `scripts/wait_for_gpu_and_run.sh` を使用します。各実験には、`RUN_SCRIPT`、`EXP_ROOT`、必要なら `SCHEDULE_LABEL` を設定して共通スクリプトを呼ぶ薄いラッパーを置いてください。

使用可能なphysical GPUはPCごとの同期対象外ファイル `~/.config/edm/gpu.env` に設定します。共有コードにGPU番号・GPU名を固定しないでください。GPU処理前に `scripts/load_gpu_config.sh` をsourceし、設定がない場合はGPU処理を開始しません。設定には `GPU_INDEX` と `EXPECTED_GPU_NAME` の両方が必要です。別パスを使う場合は `EDM_GPU_CONFIG` で指定します。

ローカル設定を優先し、コマンドラインの環境変数で許可GPUを上書きしません。読み込み処理は `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES="$GPU_INDEX"` を設定します。共通待機スクリプトはGPU名を検証し、指定GPUが使用中なら待機します。別GPUへ自動的に切り替えません。`CHECK_INTERVAL_SEC` で確認間隔を調整できます。`--help`、dry-run、計画表示、GPUを使わない前処理・集計は設定読み込み・待機対象外です。ジョブスケジューラによる排他割当がある場合も、ローカル設定のGPU制約は維持します。

例:

```bash
RUN_SCRIPT=experiments/<group>/<name>/scripts/run.sh \
EXP_ROOT=experiments/<group>/<name> \
scripts/wait_for_gpu_and_run.sh
```

## コーディング規約

Pythonは4スペースでインデントし、関数・変数には `snake_case`、クラスには `PascalCase`、モジュールには小文字名を使います。Clickのオプション名は具体的かつ後方互換にしてください。コアモジュールの区切りコメントは維持します。フォーマッターやリンターは未設定のため、無関係な整形は避け、importは標準ライブラリ、外部パッケージ、ローカルモジュールの順にまとめます。

## テスト指針

自動テストとカバレッジ基準は現在ありません。最低限、変更したPythonファイルに `python -m compileall` を実行し、関連する `--help` または `--dry-run` を確認してください。サンプラー変更では固定シードによる小規模生成、学習変更ではGPUジョブ前のドライランを行います。サンプラー、steps/NFE、チェックポイント、シード範囲、GPU数などの再現条件は実験READMEに記録してください。

## コミットとプルリクエスト

履歴にならい、`Fix a DDP consistency issue with PyTorch 2.1.` のような短く明確な要約を使い、論理的な変更単位ごとにコミットしてください。プルリクエストには目的、検証コマンド、影響する実験、関連Issueを記載します。生成処理の変更には代表的な指標や画像を添え、大容量成果物はコミットせずリンクしてください。既定値、チェックポイント互換性、データセット、GPU要件の変更は明記します。
