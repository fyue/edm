# 実験0-4：200k・210kのRestart K=1効果、FID-50k・3生成seed範囲

200k・210kでRestart K=1のFID改善が3つの生成seed範囲で再現するかをFID-50kで確かめる。このPCには初回範囲の4評価のみ（K=1の改善は200k +0.01343、210k +0.01809）で、3範囲の結果は元PCと要照合。

Slackには[2026-10-27の3生成seed範囲の報告](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793028436537769)がある（[原本・TSの索引](../../reports/slack_sources.md#s6)）。このPCの[improvement_stats.tsv](results/improvement_stats.tsv)はn=1であり、報告内容との照合は未完了。報告値で評価データを補完していない。

固定した200k・210kの各モデルで、Restart K=1のFID改善が生成seed範囲を変えても
再現するかを確認する。学習は行わず、実験0-3のモデル・コード・成果物は読み取りのみとする。
3 seedは3回の再学習ではなく、固定モデルからの3つの生成seed範囲を意味する。

## 条件

| 条件 | 通常EDM | Restart |
|---|---:|---:|
| Main steps | 18 | 18 |
| K | 0 | 1 |
| NFE | 35 | 39 |
| Restart steps | — | 3 |
| t_min / t_max | — | 0.14 / 0.30 |

共通：Heun、rho=7、sigma_min=0.002、sigma_max=80、S_churn=0、restart_noise=1、
生成/FID batch=64、1 GPU。参照統計は`fid-refs/cifar10-32x32.npz`。
クラスラベルも生成seedに対応する乱数で決定する。
K=1の実開始sigmaは約0.139516468731、restart indexは13。
main steps固定でRestartには追加4 NFEを使うため、同一NFE比較ではない。

200kは実験0-1、210kは実験0-3のEMA snapshotを使用する。対象は200k・210kのみ。
各条件50,000枚を`0-49999`、`50000-99999`、`100000-149999`で評価する。
初回範囲の4評価は0-3の完了JSONから検証して取り込み、残り8評価・40万枚を新規生成する。
初回FIDの良否による打ち切りやKの再選択は行わない。新規評価は既存実績で約10時間。

取り込み時にモデル、生成/FIDコード、0-3ランナー、FID参照統計、batch、サンプラー条件、
runtimeを照合し、元JSONと出典パスを新しい完了JSONに保持する。取り込みに元画像は不要。
新規評価は別PCのPascalでも実行できる。ローカルGPU設定から選択した実GPUのUUID・driver・
Python・PyTorch・CUDA・cuDNN・NumPy・SciPy・Pillow等を評価ごとに記録する。
0-3との環境差は警告して継続する。同じcheckpoint・seed範囲のK=0/K=1だけは同じ
実GPU・数値計算環境で揃え、不一致を拒否する。hostname・physical GPU番号・PCI bus番号の
違いだけでは拒否しない。共有コードにはGPU番号・GPU名を固定しない。

## 起動

リポジトリ直下でedm環境をactivateするか、`PYTHON=/path/to/edm/bin/python`を指定する。
torchrunは同じPythonのbinを既定とする。

```bash
RUN=experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/run.sh
bash "$RUN" --help
bash "$RUN" --plan-only

# CPUのみで既存4評価を取り込み・暫定集計
bash "$RUN" --seed-filter 0-49999

# GPU待機経由で200kのK=0/K=1を各2枚生成。FIDは行わない
bash "$RUN" --checkpoint-filter 200k --samples-only

# 全12評価を揃える。完了済み評価は検証してskip
tmux new-session -s edm-exp0-4
bash experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/schedule_after_gpu_free.sh

# または、スモーク成功後だけ本評価へ進む一括起動
bash experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/run_after_smoke.sh
```

条件境界の共通`ConditionPause`が`RUN_SCRIPT`・`EXP_ROOT`・`SCHEDULE_LABEL`を設定して
`scripts/wait_for_gpu_and_run.sh`を呼ぶ。GPUが空いたら`scripts/load_gpu_config.sh`で
ローカル設定を読み込み、実行環境を記録して同じ親runnerから評価を開始する。二重待機は行わない。
`~/.config/edm/gpu.env`（または`EDM_GPU_CONFIG`）の`GPU_INDEX`と`EXPECTED_GPU_NAME`が必要。
使用中なら待機し、別GPUへ切り替えない。確認間隔は`CHECK_INTERVAL_SEC`（既定300秒）。
help・plan・verify・aggregate・refresh-imports・既存JSONのみの取り込みはGPU待機・
設定読み込み・ETA監視を行わない。対象評価がすべて完了済みの場合も待機しない。

## 一時待機・自動再開と終了

8-13と同じ共通の一時待機ファイルを使用する。親runnerはtmux内で起動したままにする。

```bash
touch /data/kyou/STOP_RUNNING_PASCAL
# 現在のcheckpoint×K×seed範囲の生成・検証・FID・保存完了後に待機
tail -n 12 /data/kyou/STOP_RUNNING_PASCAL

unlink /data/kyou/STOP_RUNNING_PASCAL
# GPUの空きを再確認して、次の未完了評価へ自動再開
```

共通ファイルなので8-13等にも影響する。個別に制御する場合は、起動時に
`PAUSE_FILE=/absolute/path`を指定する。起動時に一時待機ファイルがあればGPU処理を
開始しない。GPU待機後にも再確認し、再作成されていたら一時待機に戻る。
評価中に作成されても生成/FIDを途中で中断せず、結果を保存する。
最後の条件まで完了した場合は一時待機せず終了する。

- `PAUSE_ENABLED=0`：一時待機とETA監視を無効化する。GPU制約・GPU空き確認と終了STOPは維持する。
- `PAUSE_POLL_SEC`：一時待機の確認間隔。既定1秒。
- `PAUSE_ETA_UPDATE_SEC`：ETA追記間隔。既定30秒。

待機・再開イベントは`logs/master.log`と`logs/pause.log`、状態は
`logs/pause-runtime.<PID>.json`、付帯処理の実測は`logs/pause-timings.jsonl`に保存する。
条件別`logs/<condition>-seeds-<range>.log`に生成・FIDの進捗を記録する。
既存の一時待機ファイルへ実験名・checkpoint・K・seed範囲・処理段階・待機開始ETAを追記する。
実測履歴不足は「推定中」とし、ファイル削除後に監視処理が再作成することはない。
スモークテストにも同じ制御を使い、対象枚数を2枚として記録する。

親runnerを終了する場合は、実験専用の終了STOPを使う。

```bash
tail -f experiments/0/4_post200k_restart_k1_fid50k_3seeds/logs/master.log

# 現在のcheckpoint×K×seed範囲を生成・FID・保存後に終了
touch experiments/0/4_post200k_restart_k1_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION

# STOPを削除し同じコマンドで再開
unlink experiments/0/4_post200k_restart_k1_fid50k_3seeds/STOP_AFTER_CURRENT_CONDITION
bash "$RUN"

# 個別選択、GPU不要の検証・再集計
bash "$RUN" --checkpoint-filter 210k --seed-filter '50000-99999,100000-149999'
bash "$RUN" --verify-only
bash "$RUN" --aggregate-only
```

終了STOPは一時待機より優先する。起動前・条件境界・GPU待機中にも確認し、
一時待機中やGPU待機中でも親runnerを終了できる。STOPは自動削除しない。
未完了生成は不足seedだけ生成し、
画像が揃っていればFIDから再開する。PNGのseed・配置・RGB・32×32を検証し、想定外画像を拒否する。
スモーク画像は`samples/`に分離し、本評価へ流用しない。完成画像を削除した場合、verifyと
通常再実行はエラーにする。完了JSONのみのaggregateは可能。

## 旧protocolからの互換性更新

待機機能追加前の0-4はランナーのハッシュが異なるため、そのままでは厳密なprotocol検証で
拒否される。初回範囲の0-3由来の取り込み結果だけが存在する場合、次で更新できる。

```bash
bash "$RUN" --refresh-imports
```

元の0-3評価・FID値・条件・protocol・出典を検証し、全件一致した場合だけ旧JSONを
`results/import_backups/<旧protocol_id>/`に退避して、新protocolで取り込み直す。
再実行は可能で、最新protocolの結果は変更しない。GPUは使わず、既存の終了STOPも保持する。
新規生成の完了結果、画像・スモーク画像・途中生成のprotocolがある場合や、条件不一致の場合は
更新を拒否する。通常起動・verifyのprotocol検証は緩めない。

## 別PCのPascalで実行する場合

最新のリポジトリコード、200k/210kのsnapshot、FID参照統計、0-3の`results/runtime.json`と
4つの初回範囲の完了JSONを同期する。モデルの配置は既存と同じリポジトリ内相対パスを維持する。
学習用dataset ZIPとoptimizer stateは評価には不要。実行するPCの`~/.config/edm/gpu.env`に
そのPCで許可されたGPUを設定する。元PCのgpu.envはコピーしない。

記録済みの絶対パスは元PCの来歴として残し、検証・画像読み込みでは現在のリポジトリ・
出力先へ読み替える。モデルと参照統計はパスではなくSHA256で実体を検証する。
既存0-4の取り込みJSONも同期する場合は、最新コードで`--refresh-imports`を一度実行する。

出力先をPCごとに分け、Pascalでは追加2範囲の両Kをまとめて実行する例：

```bash
RUN=experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/run.sh
OUT=experiments/0/4_post200k_restart_k1_fid50k_3seeds/runs/pascal

# まずK=0/K=1を各2枚確認し、成功したら次の本評価を実行する
bash "$RUN" --exp-root "$OUT" --checkpoint-filter 200k --samples-only
bash "$RUN" --exp-root "$OUT" --seed-filter '50000-99999,100000-149999'
```

各コマンドをedm環境のtmux内で実行する。上の2コマンドは個別に成功を確認して進める。
一時待機・終了STOPは出力先ごとに扱うため、専用出力先ではその配下の終了STOPを使う。
共通の一時待機ファイルは従来どおり有効。複数PCから同じ出力先を同時更新しない。

Pascal出力の`results/*-fid50000-seeds-*.json`を集計PCの同じruns/pascal/resultsへ転送後、
既存4評価と追加8評価を統合する。集計には生成画像を転送する必要はない。

```bash
bash "$RUN" --aggregate-only --result-roots \
  experiments/0/4_post200k_restart_k1_fid50k_3seeds \
  experiments/0/4_post200k_restart_k1_fid50k_3seeds/runs/pascal
```

`--result-roots`はaggregate/verify専用。同一条件・seedの重複や、同じRestart比較対の
数値計算環境不一致は統合時も拒否する。verifyは各入力出力先のローカル画像も必要とする。
集計JSON/図を書いても、別出力先の完了JSONを現在の出力先へ複製したことにはならない。

## 集計

`results/`に生FID一覧、condition_stats、paired_improvements、improvement_stats、
training_deltas、training_delta_statsをTSV/JSONで出力する。主指標は各seed範囲の
`FID(K=0) - FID(K=1)`で、正ならRestartが改善。平均、標本標準偏差（ddof=1）、
改善した範囲数を記録する。学習延長の差は`FID(210k) - FID(200k)`で正なら悪化。
`status.json`の`complete=true`は全12評価完了時のみ。n<3は暫定値、n<2の標準偏差はnull。
生FID TSV/JSONには評価ごとの実行環境も含む。`mixed_environments=true`なら複数環境での
集計であり、標準偏差には生成seed範囲と実行環境の両方の違いが含まれる。

`fid50k_3seeds.png/.svg`は生FID、Restart改善幅、200k→210k変化の3パネル。
各範囲の実測点を表示し、3範囲が揃った場合だけ平均±標本標準偏差を付記する。
過学習・統計的有意差・学習seed間のばらつきは、この実験だけでは断定しない。

## 検証

```bash
python -m compileall -q experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts
python experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/test_experiment.py -v
python experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/test_experiment.py -v
python experiments/8/4_restart_k_sweep_10k_interval/scripts/test_condition_pause.py -v
python experiments/8/4_restart_k_sweep_10k_interval/scripts/test_pause_eta.py -v
bash -n experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/*.sh
```

完了条件は12評価、4条件の平均・標準偏差、2checkpointのRestart改善幅、学習延長差、比較図が揃うこと。
コード・モデル・参照統計・評価条件の変更はprotocol不一致として拒否する。
環境差は評価ごとに記録し、同一Restart比較対の中だけ一致を要求する。
途中生成済みの同一条件を別GPUへ移して混在させることはできない。実行中は関連コードを変更しない。

### 初期確認（2026-10-23）

CPUテスト11件、compileall、シェル構文、help・plan表示が成功した。
0-3の4評価を検証して取り込み、`--verify-only`で4/12件を確認した。
暫定図の表示も確認済み。現時点では各条件n=1で、3範囲の標準偏差は未確定。

tmux `edm-exp0-4`で`run_after_smoke.sh`を起動したが、GPU待機中にユーザー指示で停止した。
終了STOPを保持している。K=0/K=1各2枚の実GPUスモークは未完了で、追加8評価も未開始。
既存の実験0-2は停止・変更していない。

### 待機機能追加時の確認（2026-10-24）

0-4のCPUテスト18件、8-13の9件、共通待機の17件、ETAの11件、計55件が成功した。
compileall・シェル構文・help・plan・verify・aggregateを確認した。
`--refresh-imports`で既存4評価の旧JSONを退避して更新し、FID値・出典・元protocolと
0-3の元結果が変わっていないことを確認した。終了STOPは保持し、GPUジョブは再起動していない。
