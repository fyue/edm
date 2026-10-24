# 実験0-4：200k・210kのRestart K=1効果、FID-50k・3生成seed範囲

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
新規評価ではローカルGPU設定から選択した実GPUのUUID・driver・Python・PyTorch・CUDA・
cuDNN・NumPy・SciPy・Pillow等を0-3のruntimeと比較し、不一致なら生成を開始しない。
共有コードにはGPU番号・GPU名を固定しない。

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

各GPU条件の前に`RUN_SCRIPT`・`EXP_ROOT`・`SCHEDULE_LABEL`を設定して共通
`scripts/wait_for_gpu_and_run.sh`を呼び、workerも`scripts/load_gpu_config.sh`をsourceする。
`~/.config/edm/gpu.env`（または`EDM_GPU_CONFIG`）の`GPU_INDEX`と`EXPECTED_GPU_NAME`が必要。
使用中なら待機し、別GPUへ切り替えない。確認間隔は`CHECK_INTERVAL_SEC`（既定300秒）。
help・plan・verify・aggregate・既存JSONのみの取り込みはGPU待機・設定読み込みを行わない。

## 再開・停止・集計

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

STOPは起動前・条件境界・GPU待機後に確認する。GPU待機中に作成した場合は、GPUが空き
workerが起動した時点で終了する。STOPは自動削除しない。未完了生成は不足seedだけ生成し、
画像が揃っていればFIDから再開する。PNGのseed・配置・RGB・32×32を検証し、想定外画像を拒否する。
スモーク画像は`samples/`に分離し、本評価へ流用しない。完成画像を削除した場合、verifyと
通常再実行はエラーにする。完了JSONのみのaggregateは可能。

`results/`に生FID一覧、condition_stats、paired_improvements、improvement_stats、
training_deltas、training_delta_statsをTSV/JSONで出力する。主指標は各seed範囲の
`FID(K=0) - FID(K=1)`で、正ならRestartが改善。平均、標本標準偏差（ddof=1）、
改善した範囲数を記録する。学習延長の差は`FID(210k) - FID(200k)`で正なら悪化。
`status.json`の`complete=true`は全12評価完了時のみ。n<3は暫定値、n<2の標準偏差はnull。

`fid50k_3seeds.png/.svg`は生FID、Restart改善幅、200k→210k変化の3パネル。
各範囲の実測点を表示し、3範囲が揃った場合だけ平均±標本標準偏差を付記する。
過学習・統計的有意差・学習seed間のばらつきは、この実験だけでは断定しない。

## 検証

```bash
python -m compileall -q experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts
python experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/test_experiment.py -v
bash -n experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/*.sh
```

完了条件は12評価、4条件の平均・標準偏差、2checkpointのRestart改善幅、学習延長差、比較図が揃うこと。
コード・モデル・環境の変更はprotocol不一致として拒否する。実行中は関連コードを変更しない。

### 初期確認（2026-10-23）

CPUテスト11件、compileall、シェル構文、help・plan表示が成功した。
0-3の4評価を検証して取り込み、`--verify-only`で4/12件を確認した。
暫定図の表示も確認済み。現時点では各条件n=1で、3範囲の標準偏差は未確定。

tmux `edm-exp0-4`で`run_after_smoke.sh`を起動した。指定GPUが実験0-2の学習に
使用されているため共通待機処理で待機中。K=0/K=1各2枚の実GPUスモークはまだ未完了で、
成功した場合だけ追加8評価へ進む。既存の実験0-2は停止・変更していない。
接続は`tmux attach -t edm-exp0-4`、進捗は`logs/master.log`で確認できる。
