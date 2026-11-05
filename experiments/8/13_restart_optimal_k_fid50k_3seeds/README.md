# 実験8-13：最適KとK=0のFID-50k・3シード比較

8-11の最適KとK=0をFID-50k×3生成seed範囲で比べる。平均改善幅は20k 0.381、60k 0.105、180k 0.019、200k 0.013。100k・140kはこのPCではn=1（Slackでは[完了と報告](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793027847627969)）。

Slack報告の原本・TSは[参照索引](../../reports/slack_sources.md#s6)から確認できる。ローカルの[improvement_stats.tsv](results/improvement_stats.tsv)との不一致は未照合であり、報告値で評価データを補完していない。

8-11で選択した最適Kを固定し、固定モデルにおける生成シード間のFID変動とpaired改善幅を確認する。学習seedは0のままであり、3回の再学習ではない。

| checkpoint (kimg) | K=0 NFE | 最適K | 最適K NFE |
|---|---:|---:|---:|
| 20k | 35 | 7 | 63 |
| 60k | 35 | 3 | 47 |
| 100k | 35 | 3 | 47 |
| 140k | 35 | 1 | 39 |
| 180k | 35 | 1 | 39 |
| 200k | 35 | 1 | 39 |

各条件で50,000枚を `0-49999`、`50000-99999`、`100000-149999` の3範囲で評価する。最初の範囲は8-11の条件別完了JSONを再利用する。合計36評価、再利用12評価、新規24評価・120万枚。Kは再選択しない。

main=18、EDM Heun、Restart steps=3、t_min=0.14、t_max=0.30、restart_noise=1、rho=7、sigma_min=0.002、sigma_max=80、S_churn=0。K=0ではRestartを使用しない。K>0のactual_t_min≈0.139516468731、restart_index=13。生成・FID batchは64、各PCで1 GPU、参照統計は `fid-refs/cifar10-32x32.npz`。モデルは8-11と同じ実験0-1のsnapshotを使い、クラスラベルは生成シードに対応する乱数で決まる。同じシード範囲をK=0と最適Kで共有する。生成CLIとサンプラーは変更しない。

## 実行

リポジトリ直下でedm環境をactivateするか、`PYTHON=/path/to/edm/bin/python` を指定する。torchrunはそのPythonのbinから選択し、`TORCHRUN`で変更できる。

```bash
RUN=experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/run.sh

# GPU不要、ファイルを書かず全36評価を表示
"$RUN" --help
"$RUN" --plan-only
"$RUN" --checkpoint-filter '180k,200k' --seed-filter '50000-99999' --plan-only

# CPUのみで既存12評価を検証・取り込み、暫定集計
"$RUN" --seed-filter 0-49999

# GPU待機経由のsmoke：20kのK=0とK=7を各2枚生成（FIDなし）
"$RUN" --checkpoint-filter 20k --samples-only --sample-seeds 0-1

# tmux内で実行。STOP確認後、ローカルGPUの空きを待って実行
"$RUN"
```

`--checkpoint-filter` は表のcheckpointをカンマまたは空白で列挙する。`--seed-filter` も3範囲から列挙する。未指定・`all`は全対象。`CHECKPOINT_FILTER`、`SEED_FILTER`、`EXP_ROOT` でも指定でき、CLIが優先する。共通待機への入口として `scripts/schedule_after_gpu_free.sh` も利用できる。

既存JSONは8-11の実験ルート（180k/200k）と `runs/20k-140k` から取得する。元の結果・protocol・出典パスを新しい完了記録へ保存し、生成・FIDコード、モデル、参照統計のSHA256、サンプラー条件とbatchを照合する。不一致は開始前にエラーにする。新ランナーのSHA256は別に記録する。既存画像は不要で、FIDの再計算もしない。

新規画像は `images/<condition>/fid-n50000-seeds-<range>/` に保存する。再実行時は記録済み結果を確認してskipし、未完了評価は不足シードのみ生成する。画像が揃っていればFIDから再開する。`--verify-only` は現在のコード・モデル・参照統計との一致と新規評価の画像を検証するため、実行したPCで使用する。

## STOP待機と自動再開

```bash
touch /data/kyou/STOP_RUNNING_PASCAL
# 現在の checkpoint × K × シード範囲 の生成・画像検証・FID・保存まで完了して待機

tail -n 12 /data/kyou/STOP_RUNNING_PASCAL

unlink /data/kyou/STOP_RUNNING_PASCAL
# 次の未完了評価の前にGPUを再確認し、自動再開
```

待機中も親runnerをtmuxなどで起動したままにする。起動時にSTOPがあればGPU処理を開始しない。全対象完了済みなら待機せず終了する。STOPが評価の途中で消えた場合も、条件境界で存在しなければ待機しない。GPU待機中に再作成されたSTOPはGPU処理の開始直前にも確認する。

GPUは `scripts/load_gpu_config.sh` と共通 `scripts/wait_for_gpu_and_run.sh` で制約する。各PCの `~/.config/edm/gpu.env`（または `EDM_GPU_CONFIG`）に `GPU_INDEX` と `EXPECTED_GPU_NAME` が必要。ローカル設定を優先し、別GPUへ切り替えない。空き確認は `CHECK_INTERVAL_SEC`（既定300秒）で行う。

- `PAUSE_ENABLED=0`：STOP待機とETA監視を無効化。GPU設定・空き確認は維持する。
- `PAUSE_FILE=/absolute/path`：監視先。既定は8-4と同じ `/data/kyou/STOP_RUNNING_PASCAL`。
- `PAUSE_POLL_SEC`：STOP検知間隔。既定1秒。
- `PAUSE_ETA_UPDATE_SEC`：ETAの定期追記間隔。既定30秒。段階変更時も追記する。

ETAには実験名、ホスト、PID、起動ID、checkpoint、K、シード範囲、段階を記録する。今回の生成進捗と直近のFID・画像検証・保存時間から推定し、履歴不足は「推定中」、超過は「見積もり超過・処理継続中」とする。STOPへの追記は排他制御し、削除後は再作成しない。symlinkや通常ファイル以外へ追記しない。追記失敗は警告を残し、待機動作を維持する。8-4と共有するETA helperは `scripts/pause_eta.py`、8-13の条件境界制御は `scripts/condition_pause.py`。

待機・再開は `logs/master.log` と `logs/pause.log`、ETA状態は `logs/pause-runtime.<PID>.json`、付帯処理の実測は `logs/pause-timings.jsonl` に保存する。help・plan・verify・aggregateはSTOP待機・GPU確認を行わない。

親runnerを終了させる場合は次を使う。両方のSTOPがあれば終了が優先する。

```bash
touch experiments/8/13_restart_optimal_k_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
```

終了用STOPを削除して同じコマンドを実行すると再開する。変更後に起動したrunnerが対象で、起動済みプロセスの動作は切り替わらない。

## 分担と集計

複数PCから同じ出力先を更新しない。例として、最初のPCが4 checkpoint、次のPCが180k/200kを担当する。

```bash
"$RUN" --checkpoint-filter '20k,60k,100k,140k' --exp-root experiments/8/13_restart_optimal_k_fid50k_3seeds/runs/early
"$RUN" --checkpoint-filter '180k,200k' --exp-root experiments/8/13_restart_optimal_k_fid50k_3seeds/runs/late

# 両PCの条件別完了JSONを各runs/*/results/に揃えてから実行
"$RUN" --aggregate-only --result-roots \
  experiments/8/13_restart_optimal_k_fid50k_3seeds/runs/early \
  experiments/8/13_restart_optimal_k_fid50k_3seeds/runs/late
```

転送には `results/*-fid50000-seeds-*.json` だけが必要。集計にモデル・画像は不要。同じ条件・シード範囲の重複は値が一致していてもエラー。シードで分担する場合も、既存範囲の取り込み担当を重複させない。集計出力を保存しただけでは他PCの評価完了マーカーにならない。

結果は `results/` に出力する。

- `restart_fid_summary.tsv/.json`：生の36評価。JSONにはprotocolと再利用元を含める。
- `condition_stats.tsv/.json`：条件別12行、FIDの平均と標本標準偏差（ddof=1）。
- `paired_improvements.tsv/.json`：各範囲の `FID(K=0) − FID(最適K)`。
- `improvement_stats.tsv/.json`：6 checkpointの改善幅の平均・標本標準偏差・正の改善となった範囲数。
- `fid50k_3seeds.png/.svg`：実測点と、3範囲が揃った場合の平均±標本標準偏差。改善幅にゼロ線を表示する。

`n`は揃った範囲数、`complete=true`はn=3。途中集計にも平均は出すが、正式な3シード結果とは扱わず、n<2の標準偏差はnull/空欄にする。欠損した範囲をゼロで埋めない。

最初の範囲はKの選択にも使用済みで、選択による楽観的な偏りを含む可能性がある。新規2範囲の改善幅も個別に確認する。3範囲だけで統計的有意差を断定しない。これは固定モデルに対する生成シード間の変動であり、学習seed間のばらつきや参照データの不確かさは評価していない。

## 検証

```bash
python -m compileall -q scripts/condition_pause.py scripts/pause_eta.py experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts
python experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/test_experiment.py -v
python experiments/8/4_restart_k_sweep_10k_interval/scripts/test_condition_pause.py -v
python experiments/8/4_restart_k_sweep_10k_interval/scripts/test_pause_eta.py -v
```

本番24評価は設定・CPU検証では起動しない。画像・モデル・ログ・大容量一時データはコミットしない。

### 初期設定時の確認（2026-10-20）

- compileall、シェル構文、help・plan表示を確認。計画は36評価中12件再利用・24件追加、120万枚。
- 新規CPUテスト9件、8-4の条件境界テスト17件・ETAテスト11件が成功。
- 既存12評価を取り込み済み。`--verify-only` と `--aggregate-only` で12/36件を確認し、暫定PNGも確認した。現在の各条件はn=1で、標準偏差は未確定。
- `--checkpoint-filter 20k --samples-only --sample-seeds 0-1` を別の一時出力先で試行したが、`nvidia-smi` がNVIDIAドライバと通信できず、共通GPU待機段階で停止した。実GPUでのK=0/K=7の生成確認は未完了。本番24評価は未起動。
