# Experiment 8-4: Restart K Sweep at 10k Checkpoint Intervals

## 目的

厳密な10k刻み再学習系列の利用可能な全checkpointで`K=0,1,...,8`を走査し、
FIDを最小にするKが学習進行に伴って変化するか調べる。checkpointは実行時に検出し、
K=0～8の条件TSVを自動生成する。checkpoint追加時にコードやTSVを編集する必要はない。

## 実験条件

- dataset/model: CIFAR-10 class-conditional EDM
- training run: `experiments/0/1_cifar10_cond_edm_10k_retraining`
- checkpoints: 実行時の指定、または保存済みの10k刻みcheckpointを自動検出
- samples: 5,000/条件
- seeds: `0-4999`（全条件で共通）
- main sampler: EDM Heun, 18 steps, `rho=7`, `S_churn=0`
- Restart: `N_restart=3`, `K=1,...,8`, `t_min=0.06`, `t_max=0.30`
- Restart noise multiplier: `1`
- K=0: Restartなしの18-step ODE
- FID reference: `fid-refs/cifar10-32x32.npz`
- GPU: 1

NFEは`35 + 4K`で、K=0から8まで`35,39,...,67`となる。1 checkpointにつき9条件。
sampler設定は[config/restart_k0_8_fid5k.tsv](config/restart_k0_8_fid5k.tsv)の先頭checkpointの
9条件をtemplateとして使い、networkとconditionを選択したcheckpointへ置き換える。

## checkpointの柔軟な指定

`CHECKPOINT_FILTER`または`CHECKPOINTS`で次の形式を指定する。

| 指定 | 対象 |
| --- | --- |
| `160k` | 1 checkpoint、9条件 |
| `120k 140k 160k` | 複数checkpoint |
| `10k-160k` | 10k刻みの範囲、両端を含む |
| `20k-160k:20k` | 20k刻みの範囲 |
| `all`または未指定 | 利用可能なcheckpointを自動選択 |

FID-5kの自動選択は保存済みの10k刻みcheckpointすべて。FID-25kの自動選択は、
そのうちFID-5kでK=0～8の結果がすべて揃ったcheckpoint。明示指定したcheckpointが
存在しない場合はエラーとし、黙って一部だけ実行しない。

新しいcheckpointでFID-25kを評価する場合は、先に同じcheckpointでFID-5kを実行する。
FID-5k画像または結果が不足している場合、FID-25k runnerは理由を表示して停止する。

```bash
CHECKPOINT_FILTER="160k" GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh

CHECKPOINT_FILTER="160k" GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_fid25k_after_gpu_free.sh
```

範囲指定の計画確認:

```bash
PLAN_ONLY=1 CHECKPOINTS="110k-160k:10k" \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh
```

checkpointディレクトリはtemplateのnetworkと同じディレクトリを使う。
`CHECKPOINT_DIR=/path/to/training-run`で変更できる。異なる学習系列を評価する場合は
`EXP_ROOT`も分け、同じcondition名の画像や結果が混ざらないようにする。
実行時に生成した条件TSVはprotocolの`logs/`へ保存し、再現条件を残す。
`PLAN_ONLY=1`の条件TSVは一時ファイルとして扱い、終了時に削除する。

以前の固定設定をそのまま使いたい場合は`CONFIG=...tsv`を明示する。この場合の
`CHECKPOINT_FILTER`は従来の空白区切りラベル指定のみで、`CHECKPOINTS`は使わない。

## 評価規則

各checkpointについて、9条件がすべて完了した場合のみFID最小のKを`optimal_k.tsv`
へ記録する。同値なら小さいKを選ぶ。FID-5kは候補選択用であり、最終値とはしない。

Kを増やすとNFEも増えるため、この実験の最適Kは「現在のRestart設定の中でFIDを
最小化するK」を意味する。同一NFEのODE比較は、選択されたK、その近傍、K=0を
FID-50kで確認する次段階に分離する。

## FID-25k拡張

当初の10k～100kの10 checkpointsを含め、同じ`K=0,...,8`走査をFID-25kへ拡張する。
各条件のseed `0-4999`は既存FID-5k画像を再利用し、seed `5000-24999`の20,000枚
だけを追加生成する。FIDは合計25,000枚に対して一度計算する。

画像は同じcondition directoryにprotocol別で保存する。

```text
images/<condition>/
  fid-n5000-seeds-0-4999/
  fid-n25000-seeds-0-24999/
```

既存5,000枚は既定でhard linkするため追加容量を消費しない。元画像と再利用画像は
immutableとして扱う。hard linkを利用できない環境では`REUSE_MODE=copy`を明示する。
結果とlogはそれぞれ`results/fid25k/`、`logs/fid25k/`へ分離する。
FID計算の直前にPNGが25,000枚ちょうどであり、相対pathがseed `0-24999`を欠損・
余剰なく表すことを検証する。同じcheckpoint・K・seedを使うため、全25,000枚を
最初から生成した場合と同じ画像集合になる。

## GPUを使わない事前確認

FID-25k拡張では、選択したcheckpointのFID-5k参照元と実行計画を確認する。画像のlink、
追加生成、FID計算は行わない。

```bash
PLAN_ONLY=1 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_fid25k_after_gpu_free.sh
```

1 checkpointだけなら9条件が表示される。

```bash
PLAN_ONLY=1 CHECKPOINT_FILTER="30k" \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_fid25k_after_gpu_free.sh
```

既存のFID-5k計画は次で確認する。

```bash
PLAN_ONLY=1 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh
```

checkpointを限定して計画を確認できる。

```bash
PLAN_ONLY=1 CHECKPOINT_FILTER="20k" \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh
```

## 実行

physical GPU 0が空くまで共通wrapperで待ってから全条件を実行する。

```bash
GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh
```

FID-25k拡張でFID-5k完了済みcheckpointをすべて実行する場合:

```bash
GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_fid25k_after_gpu_free.sh
```

FID-25k拡張をcheckpoint単位で実行する場合:

```bash
CHECKPOINT_FILTER="30k" GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_fid25k_after_gpu_free.sh
```

複数checkpointも空白区切りで指定できる。

```bash
CHECKPOINT_FILTER="20k 40k 60k" GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_fid25k_after_gpu_free.sh
```

特定checkpointだけ先行する場合:

```bash
CHECKPOINT_FILTER="10k" GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh
```

条件を限定する場合:

```bash
CONDITION_FILTER="20k-ode-k0 20k-restart-k4 20k-restart-k5 20k-restart-k6" \
  GPU_INDEX=0 \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/schedule_after_gpu_free.sh
```

進捗確認:

```bash
tail -f experiments/8/4_restart_k_sweep_10k_interval/logs/restart-k-sweep-fid5k.log
tail -f experiments/8/4_restart_k_sweep_10k_interval/logs/fid25k/restart-k-sweep-fid25k.log
```

同一条件・画像数・seed範囲・FID referenceが結果TSVに記録済みなら再実行時にskipする。

## STOPによる条件完了後の待機と自動再開

FID-5k・FID-25kと、8-4を`EXP_ROOT`に指定して共通runnerを使うFID-50kで有効。
小実験は「checkpoint × K × FID件数」の1条件を指す。
8-12と同じ`/data/kyou/STOP_RUNNING_PASCAL`を各PCで使用する。

```bash
# 現在の小実験を保存してから待機させる
touch /data/kyou/STOP_RUNNING_PASCAL

# 待機解除。親runnerが次の未完了条件から自動再開する
unlink /data/kyou/STOP_RUNNING_PASCAL
```

- 実行中の画像生成・画像検証・FID計算・結果TSV保存は最後まで行う。子プロセスへ
  SIGTERM/SIGKILLは送らず、次の未完了条件へ進む直前にSTOPの存在を確認する。
- 起動時にSTOPがあればGPU処理を開始せず待機する。条件境界の確認前にSTOPを
  削除した場合は待機しない。全対象が完了している場合も待機せず通常終了する。
- 削除後はローカルGPU設定・GPU名・空きを共通GPU待機スクリプトで再確認する。
  GPU待機中にSTOPが再作成された場合も、次のGPU処理を開始する前に再び待機する。
- `PAUSE_FILE=/absolute/path/to/STOP`で監視先、`PAUSE_POLL_SEC=1`で待機時の確認間隔を
  変更できる。`PAUSE_ENABLED=0`でこの待機機能を無効化できる。他実験では既定で無効。
- 待機・再開と次の対象条件はmaster logと`LOG_DIR/pause.log`に残す。
  `PLAN_ONLY=1`はSTOP待機とローカルGPU設定読み込みを行わない。
- 待機中は親runnerを起動したままにする（tmuxなどを使用）。すでに起動済みのrunnerへの
  変更反映は保証されないため、この機能は変更後に起動したrunnerで使用する。

### STOPファイルに表示する待機開始見込み

STOPがある間、監視helperが現在の条件・段階と「小実験完了までの残り時間」を
STOPファイルへ追記する。生成・FIDを中断する処理は行わない。
既定では検知時、処理段階が変わった時、その後30秒ごとに記録する。

```bash
tail -n 12 /data/kyou/STOP_RUNNING_PASCAL
```

表示例:

```text
[2026-10-19T15:24:00+09:00] experiment=8-4 host=PC-A pid=12345 run=12345-... checkpoint=60k K=6 FID=25000
状態=画像生成中 待機開始まで=約18分 待機開始見込み=2026-10-19T15:42:00+09:00 根拠=生成進捗＋直近FID中央値
```

- 検知間隔は`PAUSE_POLL_SEC`、定期追記間隔は`PAUSE_ETA_UPDATE_SEC`（既定30秒）。
  `PAUSE_ENABLED=0`と`PLAN_ONLY=1`では監視helperも起動しない。
- 生成時間は今回の生成ログのtqdm進捗から推定する。既存画像数や前回起動の進捗を
  今回の生成済み枚数として扱わない。FID時間は同じ画像数・参照統計の直近5件の
  `fid_sec`中央値を使う。FID中は経過時間と現在の進捗も参照する。
- 検証・保存時間は今後の同PC・同評価件数・同参照統計での実測中央値を加える。
  初回などの未計測部分は表示へ明記する。進捗・履歴が不足すれば「推定中」、
  見積もりを超えても終了していなければ「見積もり超過・処理継続中」と表示する。
  GPU速度・I/Oにより変動する概算であり、待機開始時刻を保証するものではない。
- 実際の待機開始は「待機中・残り0秒」、全対象完了は「実験終了・待機不要」を記録する。
  親runnerが異常終了した場合や終了用STOPで終了した場合も状態を記録する。
- 他runnerの記録を上書きしない。ホスト名・PID・起動IDで識別し、追記を排他制御する。
  STOP削除後は追記を止め、削除・置換との競合でもSTOPを再作成しない。
  symlinkや通常ファイル以外への追記は拒否する。
- 追記できない場合はmaster logへ警告する。実験および条件完了後の待機は維持する。
  ローカル状態・付帯時間の履歴は`LOG_DIR/pause-runtime.<PID>.json`と
  `LOG_DIR/pause-timings.jsonl`に保存する。8-12の停止方法は変更しない。

従来どおり、現在の小実験完了後に親runnerを**終了**するには次を作成する。

```bash
touch experiments/8/4_restart_k_sweep_10k_interval/STOP_AFTER_CURRENT_CONDITION
```

両方のSTOPが存在する場合は終了を優先する。終了用STOPで止めた場合は、そのファイルを
削除して同じ実行コマンドを再実行する。

CPUだけの停止・再開テスト:

```bash
python experiments/8/4_restart_k_sweep_10k_interval/scripts/test_condition_pause.py -v
python experiments/8/4_restart_k_sweep_10k_interval/scripts/test_pause_eta.py -v
```

## 集計とグラフ

完走時にrunnerが自動生成する。手動で再生成する場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/summarize_results.py

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/plot_results.py

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/4_restart_k_sweep_10k_interval/scripts/create_analysis_views.py
```

overviewグラフはKごとのFID、K=0からのFID変化、学習stepと最適Kの関係を示す。
追加の分析図表は、checkpoint別のFID–K曲線、K=0との差のヒートマップ、
K=0～8を横持ちにしたTSV・Markdown表を示す。

## 出力

```text
experiments/8/4_restart_k_sweep_10k_interval/
  images/<condition>/
    fid-n5000-seeds-0-4999/
    fid-n25000-seeds-0-24999/
  logs/
    fid25k/
  results/restart_fid_summary.tsv
  results/restart_k_sweep.tsv
  results/optimal_k.tsv
  results/restart_k_sweep_overview.{png,svg}
  results/fid_by_checkpoint_and_k.{tsv,md}
  results/fid_vs_k_by_checkpoint.{png,svg}
  results/fid_delta_vs_k0_heatmap.{png,svg}
  results/fid25k/
    reuse_manifest.tsv
    restart_fid_summary.tsv
    restart_k_sweep.tsv
    optimal_k.tsv
    fid_by_checkpoint_and_k.{tsv,md}
    restart_k_sweep_overview.{png,svg}
    fid_vs_k_by_checkpoint.{png,svg}
    fid_delta_vs_k0_heatmap.{png,svg}
```

生成画像、checkpoint、logはcommitしない。
