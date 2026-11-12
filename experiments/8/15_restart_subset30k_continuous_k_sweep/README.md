# 実験8-15：30,000枚の連続学習モデルによるRestart K走査

0-6の20k〜100k checkpointでmain18・nfe55のK=0〜8を直接FID-50kで評価し、0-2の再開による乱数再生と混在していた8-14の傾向を調べ直す。2026-11-12時点のローカル保存結果はnfe55が33/45件（20k・40k・60kは全K、80kはK=0〜5）、main18は0/45件で、全評価の完了は未確認。

要件の出典：[fyue/edm Issue #1](https://github.com/fyue/edm/issues/1)。生成・検証・集計処理は[8-14](../14_restart_subset30k_k_sweep/README.md)を基にし、出力とprotocolは8-15専用とする。8-14の画像・FID・互換登録を流用しない。

## 再現条件

| 項目 | main18 | nfe55 |
|---|---|---|
| checkpoint（kimg） | 20k・40k・60k・80k・100k | 同左 |
| K | 0〜8 | 0〜8 |
| N_main | 18 | 28−2K |
| NFE | 35+4K | 55 |

K=0はEDM Heun、K≥1はRestart Heun。N_restart=3、t_min=0.14、t_max=0.30、restart_noise=1、rho=7、sigma_min=0.002、sigma_max=80、S_churn=0。実開始sigma/indexは主スケジュールへの最近傍対応として記録する。nfe55では主スケジュールも変化する。

1 GPU、生成/FID batch=64、生成seed 0–49999、seedに対応するランダムクラスラベル、全CIFAR-10参照統計 `fid-refs/cifar10-32x32.npz` を使う。FID-5k/25kは実施しない。両系列は90条件・90回のFID-50k、本評価の最大生成数450万枚。小規模確認は各条件seed 0–1（別保存、最大180枚）。K=5も系列ごとに独立して生成・評価する。

## 起動と単独実行

リポジトリ直下でedm環境のPythonを指定する。

```bash
EXP=experiments/8/15_restart_subset30k_continuous_k_sweep
export PYTHON=/home/kyoukinkin/data/conda-envs/edm/bin/python
# GPU設定・checkpoint不要、書き込みなし
"$EXP/scripts/run_all.sh" --plan-only
"$EXP/scripts/run.sh" --help
# checkpoint監視を即開始。GPUが必要な条件の直前に共通待機処理を実行
"$EXP/scripts/run_all.sh"
# 片方だけを監視・実行（45条件・225万枚）
"$EXP/scripts/run.sh" --mode main18
"$EXP/scripts/run.sh" --mode nfe55
# 対象限定、監視間隔変更
"$EXP/scripts/run.sh" --mode nfe55 --checkpoint-filter '20k,40k' --checkpoint-poll-sec 60
# CPU専用検証・再集計
"$EXP/scripts/run_all.sh" --verify-only
"$EXP/scripts/run_all.sh" --aggregate-only
```

`--exp-root`は実験全体の出力ルートを指定し、単独実行でも系列別の出力先は `<root>/runs/main18`・`<root>/runs/nfe55`。一括実行は全体ロックに加えて両系列の実行ロックを保持し、同じ出力先の単独実行を拒否する。単独系列どうしは別々に起動できるが、同じGPUで同時に運用する場合は一括ランナーを推奨する。ロックはcheckpoint・STOP・GPU待機中も保持する。

GPU制御は共通の `scripts/load_gpu_config.sh` と `scripts/wait_for_gpu_and_run.sh` を使用する。ローカル `~/.config/edm/gpu.env`（またはEDM_GPU_CONFIG）のGPU_INDEX・EXPECTED_GPU_NAMEを優先し、別GPUに切り替えない。0-6などが設定GPUを使用中なら待機する。CHECK_INTERVAL_SECでGPU確認間隔を変更できる。任意で `scripts/schedule_after_gpu_free.sh` を使うと、監視ランナーの起動自体もGPUが空くまで待機する。

## checkpoint到着・再開

0-6の既定保存先を直接監視し、manifestの完成は要求しない。現在利用可能なcheckpointを学習量の昇順で処理し、未到着の小さいcheckpointがあっても到着済みの大きいcheckpointは評価する。各checkpointで選択した両系列の全Kの小規模生成を検証してから、main18、nfe55の順にそれぞれK=0〜8をFID-50kで評価する。単独実行では選択系列のみを処理する。

監視間隔は `--checkpoint-poll-sec` またはCHECKPOINT_POLL_SEC、既定300秒。サイズ・mtime等が30秒以上変わらないことを確認し、CUDAを隠した別CPUプロセスでEMAモデルの読込・CIFAR-10条件を検証する。確認は監視時に行うため、実際の待機は監視間隔に依存する。読込中の更新は再待機、安定したファイルの読込失敗はエラー。GPU待機後にsnapshotが変わった場合もエラーにし、再起動後に安定確認をやり直す。

再起動は同じコマンドを実行する。完了JSON、protocol、モデル・生成/FIDコード・ランナー・監視/停止/GPU制御・参照統計のSHA256、画像seed集合とPNGを検証して完了条件をスキップする。不足seedはstagingから回復し、未生成分だけ追加する。FID失敗時は完了JSONを作らずエラー終了する。protocol不一致を自動承認しない。

各系列の `results/` に条件別完了JSON、集計TSV/JSON、FID曲線・差分ヒートマップ・最適KのPNG/SVGを条件完了ごとに出力する。全対象の完了後に最終集計して正常終了する。親ランナーを起動したまま運用し、PC再起動後のサービス自動起動は対象外。

## STOP・ログ

共通 `ConditionPause` を使用し、既定 `/data/kyou/STOP_RUNNING_PASCAL` を監視する。PAUSE_FILEまたは `--pause-file` で変更できる。作成時は現在の条件の生成・検証・FID・保存・集計まで完了し、次の未完了条件の前に待機する。checkpoint待機中にSTOPが作成されても、到着後のGPU処理を開始しない。削除後は設定GPUの名前と空きを再確認し、確認中にSTOPが再作成された場合も再待機する。

STOPへの実験名・系列・checkpoint・K・段階・進捗・ETA追記は共通pause_etaで行う。履歴不足時は推定中と表示する。PAUSE_ENABLED、PAUSE_POLL_SEC、PAUSE_ETA_UPDATE_SECは8-14と同じ。STOPファイルは削除後に再作成しない。

`<root>/STOP_AFTER_CURRENT_CONDITION`、または選択系列の `runs/<mode>/STOP_AFTER_CURRENT_CONDITION` は親ランナー全体を条件境界で正常終了させる。checkpoint待機中・GPU待機中も終了できる。削除して同じコマンドを実行すれば再開する。最後の条件完了後や全件完了済みの場合はSTOP待機を行わない。

tmuxでの運用例：

```bash
mkdir -p experiments/8/15_restart_subset30k_continuous_k_sweep/logs
tmux new-session -d -s edm-exp8-15 'env PYTHON=/home/kyoukinkin/data/conda-envs/edm/bin/python experiments/8/15_restart_subset30k_continuous_k_sweep/scripts/run_all.sh >> experiments/8/15_restart_subset30k_continuous_k_sweep/logs/watch.log 2>&1'
```

checkpoint待機・評価進捗は `logs/watch.log`、条件ごとの処理は各系列の `logs/master.log`、`logs/pause.log`、条件別ログに記録する。画像・ログ・checkpointはコミットしない。

## 0-6の実際の履歴と比較上の制約

2026-11-06、このPCで10kのsnapshotを確認した。保存済みtraining_options.jsonはtraining seed=0、subset=30,000枚・subset seed=0、目標100,000 kimgを示すが、ローカルの `logs/train.log` と最終manifestはなく、0-6の学習プロセスも確認できなかった。実際の再開履歴および学習継続状況は未確認であり、無停止の1プロセス系列であるとは断定しない。学習の新規起動・再起動はこの実験の操作に含めない。

0-6のREADMEにある10kからの再開検出は実行計画の挙動であり、実際に再開した証拠ではない。後日ログ等を入手した際に再開kimg・seed・中断を追記する。0-2と0-6の比較は学習seed各1系列・固定生成seed範囲の探索であり、再開履歴が未確認の間は乱数再生の因果効果やデータ不足の因果効果を確定しない。

## CPU検証

```bash
python -m compileall -q experiments/8/15_restart_subset30k_continuous_k_sweep/scripts
FIXED_MODE=main18 python experiments/8/15_restart_subset30k_continuous_k_sweep/scripts/test_experiment.py
FIXED_MODE=nfe55 python experiments/8/15_restart_subset30k_continuous_k_sweep/scripts/test_experiment.py
python experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/test_experiment.py PauseTests
python scripts/build_experiment_index.py --check
```

CPU模擬テストで到着順・未到着・書込途中・安定性・CPU読込失敗・直接FID-50k・不足seed再開・protocol不一致・画像不足・単独系列・全件完了・終了要求・ロック・GPU不要の計画表示を検証する。CPU mock networkで各系列K=0〜8の実NFEを確認する。共通PauseTestsでSTOP待機・削除後のGPU再確認・GPU使用中の再待機・ETA追記を確認する。

## 実装検証・起動記録（2026-11-06）

main18・nfe55それぞれ16件のCPUテスト、共通停止制御3件、compileall、bash構文、help、計画表示、GPU設定なしのverify-onlyが成功した。配置済み0-6・10k snapshotの実際のCPU読込・モデル構造も検証できた。90条件・90評価・450万枚（小規模確認180枚は別）を計画表示で確認した。

tmuxセッション `edm-exp8-15` に両系列を起動し、監視プロセスと両系列のETA監視プロセスが生存していること、20k・40k・60k・80k・100kのWAITログを確認した。監視間隔は300秒。対象checkpoint未到着のため、本評価は未開始・完了0/90。0-6の学習は操作していない。

## 保存済み結果の確認（2026-11-12）

nfe55の条件別FID-50k完了JSONは33件で、20k・40k・60kは各9件、80kはK=0〜5の6件を保存している。出典は[runs/nfe55/results/restart_fid_summary.tsv](runs/nfe55/results/restart_fid_summary.tsv)。main18の完了JSONはこのPCにはない。上記11月6日の待機記録は起動時点の記録であり、現在の保存状況とは区別する。
