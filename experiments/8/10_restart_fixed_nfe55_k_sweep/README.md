# 実験8-10：NFE=55固定で学習量とKを比較

実験0-1のCIFAR-10 class-conditional EDM（DDPM++、学習seed=0）を使い、学習量ごとに、固定計算量で主経路とRestartへの計算配分を比較する。K=5は今回の生成・集計対象から除外する。実験8-4のK=5結果の取り込みは後続作業とする。

## 再現条件

checkpointは実験0-1の20k〜200k、20k刻みの10点。ここでkは学習量kimg（1kimg=1,000枚）であり、optimizer更新回数ではない。ネットワークはconfig TSVの各`network-snapshot-*.pkl`を使用する。

| K | 0 | 1 | 2 | 3 | 4 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N_main | 28 | 26 | 24 | 22 | 20 | 16 | 14 | 12 |
| NFE | 55 | 55 | 55 | 55 | 55 | 55 | 55 | 55 |

計80条件。K=0はEDM Heun、K≥1はRestart Heun（N_restart=3、t_min=0.06、t_max=0.30、restart_noise=1）。`NFE=2N_main−1+4K=55`。

- rho=7、sigma_min=0.002、sigma_max=80、S_churn=0。モデルの対応範囲でsigmaをクリップする。
- FID-5kはseed 0〜4999。FID-25kは同じ画像集合を0〜24999へ拡張し、5000〜24999のみ追加生成する。全条件・同一ステージで共通seedを使う。ラベルは生成CLIのseedに対応するランダムクラス。
- 1 GPU、生成batch=64、FID batch=64。参照統計は`fid-refs/cifar10-32x32.npz`。
- t_minは主スケジュールの最近傍sigmaへ対応する。N_mainごとに実際の開始sigmaとindexが変わるため、`actual_t_min`と`restart_index`を計画・結果・protocolに記録する。これらは本系列のEDMPrecondの恒等round_sigmaを前提に計算する。K=0ではnull。
- FID-5kは予備評価、FID-25kは比較用。固定seed集合1組の比較であり、統計的有意差を主張しない。

## 実行

リポジトリ直下から実行する。edm環境をactivateして`python`と`torchrun`を使うか、`PYTHON=/path/to/edm/bin/python`を指定する。torchrunはそのPythonと同じbinディレクトリから選択される（`TORCHRUN`で変更可能）。

```bash
conda activate edm

# 全80条件と必要な画像数を確認。GPU不要・無書き込み
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --plan-only --fid-stage all

# 20kモデル、全8候補でseed 0〜1のスモーク生成
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --samples-only --checkpoint-filter 20k

# 全条件を5k評価
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --fid-stage 5000

# 全条件を25kへ拡張。未完了の5kも先に評価する
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --fid-stage 25000

# 両ステージを一括実行（全条件の5k、その後全条件の25k）
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --fid-stage all

# 条件を絞る。空白またはカンマ区切りで指定する
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --checkpoint-filter '20k,40k' --condition-filter '20k-k0,40k-k8' --plan-only

# 保存済み画像・完了結果の検証と、集計の再生成
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --verify-only --fid-stage all
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --aggregate-only
```

GPU番号・GPU名は共有スクリプトに固定しない。GPU実行は`run.sh`で`scripts/load_gpu_config.sh`をsourceし、共通`scripts/wait_for_gpu_and_run.sh`を経由する。`~/.config/edm/gpu.env`の`GPU_INDEX`・`EXPECTED_GPU_NAME`が必須。別ファイルは`EDM_GPU_CONFIG`で指定する。ローカル設定が環境変数より優先され、別GPUに切り替えない。GPUが使用中なら待機し、確認間隔は`CHECK_INTERVAL_SEC`で変更できる。help・plan・verify・aggregateはGPU設定読み込み・待機対象外。

CLI指定は対応環境変数より優先する。`FID_STAGE`、`CHECKPOINT_FILTER`、`CONDITION_FILTER`、`EXP_ROOT`、`CONFIG`、`FID_REF`、`GEN_BATCH`、`FID_BATCH`、`PLAN_ONLY`、`VERIFY_ONLY`、`AGGREGATE_ONLY`、`SAMPLES_ONLY`、`SAMPLE_SEEDS`、`RESUME`にも対応する。別出力先を指定する場合は、待機ログも同じ場所になるよう環境変数`EXP_ROOT`を使用する。

## 再開と結果

各条件の`images/<condition>/pool/`に最大25k枚を保持し、ステージごとのハードリンクビューをFIDへ渡す。別ファイルシステムではコピーへフォールバックする。既存画像を上書きせず不足seedだけ生成し、スモーク画像は別管理する。

resumeは既定で有効。モデル、参照統計、generate.py、fid.py、本ランナーのファイル情報と生成条件をprotocolへ保存し、不一致なら停止する。完了条件をスキップする際も画像集合を検証する。破損・余分なseedはエラー。`--no-resume`は既存画像・結果がある場合に停止する。実験8-4の画像・結果の自動取り込みは行わない。

実験ルートに`STOP_AFTER_CURRENT_CONDITION`を置くと、現在の条件・ステージの保存を終えた後、次の条件を開始せず停止する。再開時はこのファイルを取り除く。

- `results/<condition>-fid<count>.json`：完了マーカー。実測FID、seed、生成条件、実際の開始sigma、protocol ID、時間、画像・ログへのパスを保存する。
- `restart_fid_summary.tsv/.json`：完了結果一覧。
- `fid_by_checkpoint_and_k.tsv`：FIDとK=0との差（FID(K)−FID(0)、負なら改善）。
- `optimal_k.tsv`：checkpoint・ステージごとの全8候補が揃った場合のみ、候補内で最適のKを記録。同値なら小さいK。K=5を除外しているため、K=0〜8全体の最適値とは扱わない。
- `fixed_nfe55_fid{5000,25000}.{png,svg}`：FID–K曲線、K=0との差のヒートマップ、学習量と候補内最適K。

全25k評価まで新規画像は200万枚、FIDは160回。生成時間は当該ステージを完了した実行での追加生成時間であり、中断した過去の実行時間は含まない。生成画像・ログ・キャッシュ・checkpointはコミットしない。

## 検証

```bash
python -m compileall -q experiments/8/10_restart_fixed_nfe55_k_sweep/scripts
python experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/test_experiment.py
bash -n experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/schedule_after_gpu_free.sh
experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh --help
```

CPUテストは全Kのサンプラー呼び出し回数55、実際の開始sigma、追加seedとresume、protocol変更拒否、部分集計・同値の最適K、PNG破損、plan-onlyの無書き込み、GPU設定欠落時の停止を確認する。合成FIDは一時ディレクトリだけに保存する。実モデルでのGPUスモークと本評価は別途実行する。
