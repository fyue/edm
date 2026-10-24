# 実験8-12：NFE=55・t_min=0.14固定のK sweep

実験0-1のCIFAR-10 class-conditional EDM（DDPM++、学習seed=0）を使い、固定計算量で主経路とRestartへの計算配分を比較する。実験8-10のt_min=0.06固定比較、実験8-11のN_main=18固定比較とは条件が異なる。両PCの全270評価は完了し、2026-10-20に統合した。統合結果と再生成手順は[results/README.md](results/README.md)を参照。PCごとの元結果は`runs/*/results/`に記録する。

## 再現条件

- checkpoint：実験0-1の20k〜200k、20k刻みの10点。kは学習量kimg（1kimg=1,000枚）。config TSVにモデルパスを記載する。
- K=0〜8、全90条件。FID-5k/25k/50kの270評価、最大450万枚。
- `N_main=28−2K`、`NFE=2N_main−1+4K=55`。K=0はEDM Heun。K≥1はRestart Heun、N_restart=3、t_min=0.14、t_max=0.30、restart_noise=1。K=0のconfigではRestart steps・t_min/maxは0、actual_t_min/indexはnull。
- rho=7、sigma_min=0.002、sigma_max=80、S_churn=0。モデルの対応範囲でsigmaをクリップする。
- 各PCで1 GPU、生成batch=64、FID batch=64。参照統計は`fid-refs/cifar10-32x32.npz`。
- 5kはseed 0〜4999、25kは0〜24999、50kは0〜49999。全条件で同じseed集合とseedに対応するランダムクラスラベルを使用する。
- 固定seed集合1組の比較として扱い、統計的有意差は主張しない。FIDは同じ枚数のステージ内で比較する。

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N_main | 28 | 26 | 24 | 22 | 20 | 18 | 16 | 14 | 12 |
| NFE | 55 | 55 | 55 | 55 | 55 | 55 | 55 | 55 | 55 |
| actual_t_min (approx.) | — | 0.14864 | 0.10904 | 0.14489 | 0.09897 | 0.13952 | 0.08509 | 0.13120 | 0.06501 |
| restart_index (0始まり) | — | 19 | 18 | 16 | 15 | 13 | 12 | 10 | 9 |

指定t_min=0.14に最も近い主スケジュールのsigmaでRestartを開始する。N_mainが変わると実開始sigmaも変わるため、実開始sigmaまで固定した比較ではない。上表はEDMPrecondの恒等round_sigmaの場合。`actual_t_min`と`restart_index`を計画・protocol・結果に記録する。

## 実行とPC分担

リポジトリ直下でedm環境をactivateして実行する。別Pythonは`PYTHON=/path/to/edm/bin/python`を指定する。torchrunはそのPythonと同じbinディレクトリから選ぶ（`TORCHRUN`で変更可能）。

```bash
conda activate edm
RUN=experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/run.sh

# 無書き込み、GPU不要：90条件・270評価・450万枚を確認
"$RUN" --plan-only --fid-stage all

# 本評価前：各PCが担当する最初のcheckpointで全K・seed 0〜1を生成・検証
"$RUN" --samples-only --checkpoint-filter 20k --sample-seeds 0-1
"$RUN" --samples-only --checkpoint-filter 140k --sample-seeds 0-1

# RTX 8000側：6 checkpoint、54条件、162評価、270万枚
EXP_ROOT=experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k \
  "$RUN" --checkpoint-filter '20k-120k' --fid-stage all

# Pascal側：4 checkpoint、36条件、108評価、180万枚
EXP_ROOT=experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/140k-200k \
  "$RUN" --checkpoint-filter '140k-200k' --fid-stage all

# ステージ指定：未完了の前段階を含め、指定枚数まで実行
"$RUN" --checkpoint-filter 20k --fid-stage 5000
"$RUN" --checkpoint-filter 20k --fid-stage 25000
"$RUN" --checkpoint-filter 20k --fid-stage 50000

# 保存済み画像・protocol・結果の検証、集計
"$RUN" --verify-only --fid-stage all
"$RUN" --aggregate-only

# 両出力先の条件別完了JSONを同じPCへ集めた後、統合集計
"$RUN" --aggregate-only --result-roots \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k \
  experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/140k-200k
```

スモークも各PCの担当`EXP_ROOT`を指定すれば、同じ出力先で検証できる。スモーク画像は本評価poolとは別管理する。本評価は全選択条件の5k→全選択条件の25k→全選択条件の50kの順。

checkpointは単一・列挙・範囲・間隔付き範囲を指定可能（例：`40k`、`20k,60k`、`20k-120k`、`20k-140k:40k`）。未指定または`all`で全10点。間隔は正の20kの倍数、範囲は両端を含む。条件は`--condition-filter '20k-k0,20k-k8'`で絞れる。選択したモデルと参照統計だけ存在・SHA256を確認する。planはモデルを必要とするが書き込み・GPU設定は不要。

CLIは対応環境変数より優先。`FID_STAGE`（既定5000）、`CHECKPOINT_FILTER`、`CONDITION_FILTER`、`EXP_ROOT`、`CONFIG`、`FID_REF`、`GEN_BATCH`、`FID_BATCH`、`PLAN_ONLY`、`VERIFY_ONLY`、`AGGREGATE_ONLY`、`SAMPLES_ONLY`、`SAMPLE_SEEDS`、`RESUME`、`PAUSE_FILE`、`PAUSE_POLL_SEC`に対応する。

各PCのローカル`~/.config/edm/gpu.env`のGPU_INDEX・EXPECTED_GPU_NAMEが必須（別設定はEDM_GPU_CONFIG）。ローカル設定を優先し、GPU番号・名前は共有コードに固定しない。GPU処理は共通`scripts/wait_for_gpu_and_run.sh`で指定GPUの空きを待つ。別GPUへ自動切替しない。GPU待機間隔はCHECK_INTERVAL_SEC（共通既定300秒）。help・plan・verify・aggregateはGPU設定不要。

## 停止ファイルと自動再開

両PCで、それぞれの絶対パス`/data/kyou/STOP_RUNNING_PASCAL`を監視する。名前はGPU選択に使わない。両方を止めるには両PCにファイルを置く。

- `--pause-file`（環境変数PAUSE_FILE）で変更可能。`--pause-poll-sec`（PAUSE_POLL_SEC）の既定は1秒。
- 起動時にファイルがあればGPU処理を開始せず待機する。
- 実行中に検知したら、当ランナーが起動した生成・FIDのプロセスグループへSIGTERMを送る。10秒以内に終了しない場合はSIGKILLする。通常は監視間隔以内に検知するが、終了・画像検証・保存には追加時間が必要。
- 親ランナーは正常画像と進捗を保存して待機する。待機中も`logs/.run.lock`を保持し、同じ出力先の二重実行を拒否する。
- ファイル削除後、ローカルGPU設定・GPU名・空きを共通待機スクリプトで再確認し、同じ条件・ステージから自動再開する。親ランナーは起動したままにする（tmuxなどを使用）。
- GPU待機中にファイルが置かれた場合も、GPUが空いた後に再確認し、生成・FIDを開始しない。
- 停止・終了・保存枚数・待機・再開は`logs/pause.log`、子プロセス出力は条件別ログに保存する。停止ファイル以外の子プロセス失敗はエラー終了し、無限再試行しない。

従来の`EXP_ROOT/STOP_AFTER_CURRENT_CONDITION`は現在の条件・FIDステージ完了後に親ランナーを終了する操作として残す。再開にはこのファイルを取り除き、同じコマンドを再実行する。

## 保存、再開、集計

生成は条件別`pool-staging/`へ書き込む。正常終了・中断・次回起動時にRGB・32×32・PNG形式を検証し、正常なseedだけをハードリンクで原子的にpoolへ公開して一時画像を削除する。一時保存先の不完全PNGは削除し再生成する。本画像集合に既に存在する画像を上書きせず、異なる画像なら停止する。本画像集合の破損・余分なseedはエラー。

各条件の`images/<condition>/pool/`は最大50k枚。25kでは5000〜24999、50kでは25000〜49999を追加する。生成中断後は不足seedだけ再生成する。FID中断後は画像を保持し、FID計算を最初からやり直す。ステージ別ビューはハードリンク、別ファイルシステムではコピー。画像・FIDは他実験から取り込まない。

protocolはモデル・参照統計・generate.py/fid.py/ランナーのSHA256とサンプラー・seed・batchを記録する。変更時は混在させず停止する。resumeは既定有効。`--no-resume`は既存データがあれば停止する。

- `results/<condition>-fid<count>.json`：正常終了かつ有効FID確認後の完了マーカー。protocol・実開始sigma/index・GPU名・生成枚数・時間を含む。
- `progress/<condition>-fid<count>.json`：中断を含む生成・FID処理時間と保存枚数。待機時間を除く。protocol・進捗・完了JSONは一時ファイルから原子的に置換する。
- `restart_fid_summary.tsv/.json`：完了結果一覧。
- `fid_by_checkpoint_and_k.tsv`：FIDとFID(K)−FID(0)。負なら改善。
- `optimal_k.tsv`：9候補が揃ったcheckpoint・ステージのみ最適Kを記録。同値なら小さいK。
- `fixed_nfe55_tmin014_fid{5000,25000,50000}.{png,svg}`：FID–K、差分ヒートマップ、学習量と最適K。
- `time_estimate.json`：各PCの担当条件の残り時間見積もり。

結果転送は条件別完了JSONだけでよい。転送先も`<出力先>/results/`構造を保つ。統合はprotocolと同一checkpointのモデルSHA256を照合し、PC固有パス・GPU名の差を許容する。同一条件・ステージの重複はエラー。

## 8-11と同じ形式で結果図を再描画

専用の`plot_results.py`で保存済み集計JSONから図だけを再生成する。途中結果にも対応し、モデル・GPUは不要。結果JSON・TSV、実行中のランナー、protocolは変更しない。

```bash
python experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py \
  --summary experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k/results/restart_fid_summary.json \
  --outdir experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/runs/20k-120k/results

# 統合後：既定では実験直下results/の集計JSONを使い、同じ場所に出力
python experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/plot_results.py
```

8-11と同じ16×10の2×2パネル、PNG/SVG・180dpiとする。左上にFID–K曲線、右上に差分ヒートマップ、左下に学習量と最適K、右下にK=0のFIDとRestart改善量の散布図を配置する。全体タイトルに実験番号・FID枚数・NFE=55・t_min=0.14を表示し、FID曲線の凡例は上部5列に配置する。Kの目盛りは0〜8、最適K図の学習量は20k〜200kの20k刻み。checkpointの線色は欠測によらず固定する。

散布図の横軸は`FID(K=0)`、縦軸は`FID(K=0)−FID(K)`（正なら改善）。K=1〜8を8-11と同じ色で区別し、0の破線と薄いグリッドを付ける。全取得ステージの有効な改善量に0を含め、上下に範囲幅の5％の余白を付けた共通縦軸を使う。K=0または対象Kがない点は描画しない。単独散布図は出力しない。

ヒートマップは`FID(K)−FID(0)`を青（改善）→純白（0）→赤（悪化）で表示する。色範囲は−0.4〜+0.4固定、カラーバーは0.1刻み。範囲外は端の色に飽和させ、超過した側に三角マークを付ける。元の差分値は変更しない。未取得セルは薄い灰色とし、K=0が未取得のcheckpointでは差分を表示しない。最適Kは全9条件が揃ったcheckpointだけ表示し、同値なら小さいKを選ぶ。

ランナーの自動描画は従来の形式のままなので、評価・集計後に上記コマンドを再実行して表示を揃える。再描画は実行時点の集計JSONに含まれるステージだけが対象。

## 所要時間

実験8-11の保存済み実測をGPU別に集計した中央値から、生成時間を画像数×NFE、FIDと画像処理時間を評価枚数に比例させた概算。5k→25k→50kの段階評価を含む、開始からの累計。

| 担当PC | 5kまで | 25kまで | 50kまで |
|---|---:|---:|---:|
| RTX 8000：20k〜120k | 約11時間 | 約55時間 | 約114時間（4.7日） |
| TITAN X (Pascal)：140k〜200k | 約12時間 | 約59時間 | 約119時間（5.0日） |

2台同時なら全体約5日、運用上は5〜6日＋停止・GPU待機時間を見込む。中断したFIDの再実行、初回準備、転送は含めない。推定の基礎はRTX 8000の生成約0.002429秒/画像/NFE・FID約0.010861秒/画像、Pascalの生成約0.004250秒/画像/NFE・FID約0.002638秒/画像。I/O・負荷で変動する。

各PCの最初のcheckpointの全K・FID-5k完了後、実測から残り時間を更新する。過去の中断時間は結果の処理時間に含めるが、残り時間は中断・待機を予測しない。

## 検証

```bash
python -m compileall -q experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts
python experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/test_experiment.py
bash -n experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/run.sh
bash -n experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/schedule_after_gpu_free.sh
experiments/8/12_restart_fixed_nfe55_tmin014_k_sweep/scripts/run.sh --help
```

CPUテストは実NFE=55、開始sigma/index、分担件数、段階追加生成、protocol不一致、部分・統合集計、PNG破損、起動前待機、ダミー子プロセスの中断と強制終了、不足seed再開、FID再実行、planの無書き込みを検証する。GPUスモーク・本評価は別途実行する。生成画像・ログ・キャッシュ・進捗・checkpointはコミットしない。
