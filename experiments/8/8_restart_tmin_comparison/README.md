# 実験8-8：Restart t_min 比較（K=5）

EDM条件付きCIFAR-10の正確な20k・40k・60kチェックポイントで、Restart区間 `(0.06, 0.30)` と `(0.14, 0.30)` を比較する。同じ画像集合をFID-5k→25k→50kへ段階的に拡張する。

## 再現条件

| checkpoint | t_min | t_max | K | Restart steps | Main steps | NFE |
|---|---:|---:|---:|---:|---:|---:|
| 20k | 0.06 | 0.30 | 5 | 3 | 18 | 55 |
| 20k | 0.14 | 0.30 | 5 | 3 | 18 | 55 |
| 40k | 0.06 | 0.30 | 5 | 3 | 18 | 55 |
| 40k | 0.14 | 0.30 | 5 | 3 | 18 | 55 |
| 60k | 0.06 | 0.30 | 5 | 3 | 18 | 55 |
| 60k | 0.14 | 0.30 | 5 | 3 | 18 | 55 |

チェックポイントは実験0-1の `cifar10-cond-ddpmpp-edm-seed0-10k/network-snapshot-{020000,040000,060000}.pkl` を使用する。両t_min条件で同じチェックポイントとseedを使う。

- サンプラー: `generate.py --sampler=restart`、EDM Heun、rho=7。
- sigma_min=0.002、sigma_max=80（モデルの対応範囲でクリップ）、S_churn=0、restart_noise=1。クラスラベルはseedに対応するランダムラベル。
- NFE: `(2 × 18 − 1) + 5 × 2 × (3 − 1) = 55`。
- FID-5k: seed `0–4999`。FID-25k: seed `0–24999`、追加生成は `5000–24999`。
- FID-50k: seed `0–49999`、25kからの追加生成は `25000–49999`。
- 1 GPU、生成batch=64、FID batch=64。参照統計は `fid-refs/cifar10-32x32.npz`。
- 各FIDは単一の固定seed集合による簡易比較。FID-5kとFID-25kの絶対値はサンプル数の影響を受けるため、同じステージの条件間で比較する。統計的有意差は主張しない。

## RestartとEDMの時刻

公式RestartとEDM実装はいずれもこの箇所では `t = sigma` であり、追加変換は不要。公式Restartはsigma_min/maxからEDMのrhoスケジュールを作り、t_minに最も近い主スケジュールのノイズレベルへマップする。これは連続時間VP-SDEの時刻そのものではない。参照: [公式Restart実装](https://github.com/Newbeeer/diffusion_restart_sampling/blob/main/benchmarks/generate_restart.py)、[公式EDM実装](https://github.com/NVlabs/edm/blob/main/generate.py)。

この設定の主スケジュールではt_min=0.06はindex 14（0始まり）の約0.05994731、t_min=0.14はindex 13の約0.13951647となる（EDMPrecondの恒等round_sigmaの場合）。Restartの前向きジャンプの分散は `t_max² − actual_t_min²`。コアの `generate.py` と時刻変換は既存の実装を使用する。

[公式EDM向けパラメータ](https://github.com/Newbeeer/diffusion_restart_sampling/blob/main/benchmarks/params_cifar10_edm.txt) は `18 {"0": [3, 2, 0.14, 0.30]}`。今回のKは両条件で5に固定し、t_minの影響を比較する。公式設定の完全再現にはKを含め別の比較が必要。

## 実行

リポジトリ直下で実行する。既定Pythonは `/home/kyoukinkin/data/conda-envs/edm/bin/python`。`PYTHON` と `TORCHRUN` で変更できる。

```bash
# 読み取り専用の計画・入力検証（GPU不要）
experiments/8/8_restart_tmin_comparison/scripts/run.sh --plan-only --fid-stage all

# 固定seed 2枚 × 6条件のスモークテスト
experiments/8/8_restart_tmin_comparison/scripts/run.sh --samples-only --sample-seeds 0-1

# まず5k
experiments/8/8_restart_tmin_comparison/scripts/run.sh --fid-stage 5000

# 5k完了条件をスキップし、25kへ拡張（5k未完了なら先に評価）
experiments/8/8_restart_tmin_comparison/scripts/run.sh --fid-stage 25000

# 既存の5k/25k結果を検証して保持し、50kへ拡張
experiments/8/8_restart_tmin_comparison/scripts/run.sh --fid-stage 50000

# 条件を絞る。フィルターは空白またはカンマ区切り
experiments/8/8_restart_tmin_comparison/scripts/run.sh --checkpoint-filter 20k --fid-stage all

# 既存画像・protocol・完了ステージの検証のみ
experiments/8/8_restart_tmin_comparison/scripts/run.sh --verify-only --fid-stage all

# 完了した条件の比較表・グラフを再構築（GPU不要）
experiments/8/8_restart_tmin_comparison/scripts/run.sh --aggregate-only
```

`FID_STAGE`、`CHECKPOINT_FILTER`、`CONDITION_FILTER`、`PLAN_ONLY`、`SAMPLES_ONLY`、`SAMPLE_SEEDS`、`VERIFY_ONLY`、`AGGREGATE_ONLY`、`RESUME`、`EXP_ROOT`、`CONFIG`、`FID_REF`、`GEN_BATCH`、`FID_BATCH`にも対応する。CLI指定が優先。CLIの `--exp-root` でGPU待機ログも同じ場所に置く場合は、環境変数 `EXP_ROOT` にも同じパスを指定する。

`run.sh` はGPU処理を `schedule_after_gpu_free.sh` → 共通 `scripts/wait_for_gpu_and_run.sh` 経由で開始する。physical GPU 0だけを使用し、GPU_INDEX=0、CUDA_DEVICE_ORDER=PCI_BUS_ID、CUDA_VISIBLE_DEVICES=0を設定する。他のGPU指定は起動前に拒否する。EXPECTED_GPU_NAME="Quadro RTX 8000"。待機間隔はCHECK_INTERVAL_SEC（既定300秒）。GPU 0が使用中なら待機する。`--help`、plan-only、verify-only、aggregate-onlyは待機対象外。

```bash
GPU_INDEX=0 CHECK_INTERVAL_SEC=30 \
experiments/8/8_restart_tmin_comparison/scripts/run.sh --samples-only
```

長時間評価はtmuxセッション `edm-exp8-8` 内で `GPU_INDEX=0 CHECK_INTERVAL_SEC=30 experiments/8/8_restart_tmin_comparison/scripts/run.sh --fid-stage 50000 --resume` を実行する。`tmux attach -t edm-exp8-8` で接続できる。中断後はstop-fileを取り除き、同じコマンドで既存画像・完了結果を保持して再開する。CPUのみの完了監視も同セッションの別ウィンドウで実行する。

## 画像再利用とresume

実験8-8にある画像を優先して保持する。0.06条件の補完候補は実験8-6の `20k/40k/60k-restart-k5` の最大記録済みステージ、次に実験8-4の5k画像とする。`--progressive-source-root` と `--source-root` で候補ルートを指定できる。

summaryとconfigのcheckpoint、main steps、Restart steps、K、t_min/max、NFE、モデルファイルを検証し、記録されたseed範囲内のPNG整合性、RGB・32×32をseedごとに確認する。欠損・破損した候補画像は取り込まず、有効画像だけを再利用する。候補自体の設定不一致は候補全体を拒否する。実験8-4/8-6にはprotocol JSONがないため、rho=7、restart_noise=1、batch=64は既存ランナーの既定設定として扱い、この制約を取り込みメタデータ `.imports.json` に保存する。生成batchを64から変更した場合はこれらの画像を再利用しない。

検証済み画像はハードリンクで取り込む。別ファイルシステムなどでリンクできない場合はコピーする。再利用元のFID値は転記せず、60kも実験8-8内で再評価する。実験8-4/8-6のファイルを上書き・削除しない。

各条件は `images/<condition>/pool/` に最大50k枚を保持する。FIDには5k/25k/50kそれぞれ専用のハードリンクビューを渡し、fid.pyによるランダムな部分集合の選択を防ぐ。欠けたseedだけを生成するため、拡張・中断後の再開で既存画像を上書きしない。samples-only画像は別ディレクトリで管理する。

resumeは既定で有効。モデル・参照統計・generate.py/fid.pyのファイル情報とサンプラー・batch設定をprotocol JSONに記録し、変更時は既存画像と混ぜずエラーにする。`--no-resume` は既存データがあると失敗する。完了結果をスキップする際もprotocolと画像集合を検証する。実験8-8内の破損・余分なseedはエラーとして扱う。

基底 `.protocol.json` のseed範囲 `0–24999` とprotocol IDは維持する。50kへの追加範囲 `25000–49999` は `.extension.json` に基底protocol IDと最大画像数を記録する。25kを超える画像には対応する拡張メタデータが必要。既存の5k/25k結果JSON・画像ビューのprotocol IDを変更しない。

`--fid-stage 50000` と `all` は全3ステージを順番に処理する。5000/25000指定の到達点は従来どおり。今回の拡張前は20k/40kの4条件に各25k枚・結果8件があり、60kの0.06条件は実験8-6から25k枚を再利用可能。これらを検証して使えれば、新規生成は175k枚、追加FID評価は10件となる。

`STOP_AFTER_CURRENT_CONDITION` を実験ルートに作成すると、現在の条件の生成・FID・保存を終えて次の条件の開始前に停止する。起動時に存在すればGPUコマンドを開始しない。再開時はこのファイルを取り除く。

## 結果と検証

`results/<condition>-fid<count>.json` が完了マーカー。これから `restart_fid_summary.tsv/.json` と `tmin_comparison.tsv/.json` を再構築する。比較の `delta = FID(0.06) − FID(0.14)` は正なら0.14が良い。片方未完了の対はNAで記録し、完成した対についてステージ別PNG/SVGのFID棒グラフと差分グラフを保存する。

生結果にはseed範囲、ネットワーク、NFE、protocol ID、画像ビュー、ログ、画像再利用元、生成秒数・FID秒数を含む。生成秒数は結果を完了させた実行中の生成コマンドの合計であり、前回の中断までの時間・リンク時間は含まない。

ステージ別の `results/restart_tmin_fid{5000,25000,50000}_lines.{png,svg}` は、横軸20k・40k・60k、縦軸FIDの折れ線図。t_min=0.06と0.14を各1本の線で表示し、各点に実測FID値を付記する。`--aggregate-only` でも再生成できる。

長時間実行の完了チェックは、GPUを使わない `scripts/watch_completion.py` が行う。実行中の状態を `results/completion.json` に記録し、ランナー終了後に全18件の結果、各50k画像、5k/25k/50kの専用ビューとpoolの一致、基底protocol/拡張メタデータ、各折れ線図の3チェックポイントと2条件ラベルを検証する。`extension_baseline.json` のSHA256で拡張前8件の結果保持も確認する。成功時はstatus=complete、未完了でランナーが終了した場合はfailedまたはstoppedを記録する。実験の再起動は行わない。

```bash
# 本評価を別プロセスで実行中に開始するCPUのみの完了監視
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/8/8_restart_tmin_comparison/scripts/watch_completion.py

# 評価終了後の完了確認を再実行
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/8/8_restart_tmin_comparison/scripts/watch_completion.py --verify-now
```

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python -m compileall -q experiments/8/8_restart_tmin_comparison/scripts
/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/8/8_restart_tmin_comparison/scripts/test_experiment.py
bash -n experiments/8/8_restart_tmin_comparison/scripts/run.sh experiments/8/8_restart_tmin_comparison/scripts/schedule_after_gpu_free.sh
```

テストでは25k/50kへの追加seed範囲・resume（GPUコマンドを模擬）、既存protocol IDの維持、ハードリンクとコピー、有効seedだけの取り込み、protocol/extension変更拒否、stop-file、plan-onlyの無書き込み、比較差分の符号、3ステージ・3チェックポイントのグラフ出力を検証する。テストの合成FIDは一時ディレクトリだけに保存する。中止したCPU所要時間計測は再開しない。

生成画像・ログ・キャッシュはGit対象外。コンパクトな実測JSON/TSVと結果グラフだけを共有する。
