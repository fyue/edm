# Repository Guidelines

## プロジェクト構成

主要なCLIはリポジトリ直下にあります。`train.py` は学習、`generate.py` は画像生成、`fid.py` はFID評価、`dataset_tool.py` はデータセット変換を担当します。モデル、損失、拡張、データセット、学習ループは `training/`、共通機能は `torch_utils/` と `dnnlib/` に配置します。再利用する自動化は `scripts/`、実験固有の設定・スクリプト・結果・記録は `experiments/<グループ>/<実験名>/` にまとめてください。CLI資料と画像は `docs/` に置きます。生成画像、チェックポイント、ダウンロード済みモデル、大容量のFID一時データはコミットしないでください。

## ビルド・テスト・開発コマンド

- `conda env create -f environment.yml -n edm && conda activate edm`: Python 3.8/3.9環境を作成します。
- `python example.py`: 学習済みモデルのデモを実行します。GPUとネットワーク接続が必要です。
- `python generate.py --help`、`python train.py --help`、`python fid.py --help`: CLIのインポートとオプションを確認します。
- `python train.py --outdir=/tmp/edm-dry-run --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --dry-run`: 学習を開始せず設定を検証します。
- `docker build -t edm:latest .`: NVIDIA PyTorch開発イメージを作成します。

## GPU実験の実行

GPUを使う学習、生成、FID評価などの実験スクリプトやプログラムは、原則としてGPUが空くまで待機してから開始する処理を備えてください。待機処理を実験ごとに複製せず、共通の `scripts/wait_for_gpu_and_run.sh` を使用します。各実験には、`RUN_SCRIPT`、`EXP_ROOT`、必要なら `SCHEDULE_LABEL` を設定して共通スクリプトを呼ぶ薄いラッパーを置いてください。

対象GPUはphysical indexの `GPU_INDEX`（既定値 `0`）で指定し、共通スクリプトが同じ値を `CUDA_VISIBLE_DEVICES` に設定します。必要に応じて `EXPECTED_GPU_NAME` でGPU名を検証し、`CHECK_INTERVAL_SEC` で確認間隔を調整します。`--help`、dry-run、計画表示、GPUを使わない前処理・集計は待機対象外です。既にジョブスケジューラからGPUを排他的に割り当てられている場合も、二重スケジューリングを避けるため対象外にできます。

例:

```bash
RUN_SCRIPT=experiments/<group>/<name>/scripts/run.sh \
EXP_ROOT=experiments/<group>/<name> \
GPU_INDEX=0 \
scripts/wait_for_gpu_and_run.sh
```

## コーディング規約

Pythonは4スペースでインデントし、関数・変数には `snake_case`、クラスには `PascalCase`、モジュールには小文字名を使います。Clickのオプション名は具体的かつ後方互換にしてください。コアモジュールの区切りコメントは維持します。フォーマッターやリンターは未設定のため、無関係な整形は避け、importは標準ライブラリ、外部パッケージ、ローカルモジュールの順にまとめます。

## テスト指針

自動テストとカバレッジ基準は現在ありません。最低限、変更したPythonファイルに `python -m compileall` を実行し、関連する `--help` または `--dry-run` を確認してください。サンプラー変更では固定シードによる小規模生成、学習変更ではGPUジョブ前のドライランを行います。サンプラー、steps/NFE、チェックポイント、シード範囲、GPU数などの再現条件は実験READMEに記録してください。

## コミットとプルリクエスト

履歴にならい、`Fix a DDP consistency issue with PyTorch 2.1.` のような短く明確な要約を使い、論理的な変更単位ごとにコミットしてください。プルリクエストには目的、検証コマンド、影響する実験、関連Issueを記載します。生成処理の変更には代表的な指標や画像を添え、大容量成果物はコミットせずリンクしてください。既定値、チェックポイント互換性、データセット、GPU要件の変更は明記します。
