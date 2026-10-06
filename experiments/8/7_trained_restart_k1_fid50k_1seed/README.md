# Experiment 8-7: Trained Restart K=1 vs. Equal-NFE ODE, FID-50k

## 目的と実験条件

十分学習した2 checkpointについて、Restart K=1と同計算量の通常ODEを比較する。
各条件のFID-50kをseed範囲`0-49999`の1組で評価する。「1 seed」は1枚の生成ではなく、
50,000個の生成seedからなる1範囲を意味する。seed間平均・標準偏差は算出しない。

| checkpoint ID | checkpoint |
| --- | --- |
| `exp3-best` | 実験3-10の`network-snapshot-170000.pkl` |
| `pretrained` | EDM公式の`edm-cifar10-32x32-cond-vp.pkl` |

実験3のcheckpointは、既存の3-seed FID平均で最良のものを固定する。
[実験3-10のaggregate](../../3/10_160k-fid2.05-to-170k-lr2e-4-ema0.5/results/aggregate.tsv)で
170kのFID平均は`1.85124`。この既存値は18-step ODE（35 NFE）の値であり、
今回の39 NFEの対照としては使わない。両条件を新しく評価する。
checkpoint pathと公式URLは[条件TSV](config/restart_k1_fid50k.tsv)に記載する。

| 条件 | main steps | N_restart | K | t_min / t_max | NFE |
| --- | ---: | ---: | ---: | --- | ---: |
| Restart K=1 | 18 | 3 | 1 | 0.06 / 0.30 | 39 |
| 通常ODE（EDM Heun） | 20 | — | 0 | — | 39 |

RestartのNFEは`2*18-1 + 2*1*(3-1) = 39`、ODEは`2*20-1 = 39`。
両条件の初期latentとclass labelには同じseedを用いる。
Restartの`t_min`は実装に従い18-step main schedule上の最寄りのsigmaに対応する。

- dataset/model: CIFAR-10 class-conditional EDM、保存checkpointのEMA network
- samples: 50,000/条件、seeds: `0-49999`
- `rho=7`, `S_churn=0`, Restart noise multiplier `1`
- sigma設定は既存samplerの既定値（`sigma_min=0.002`, `sigma_max=80`、network対応範囲に制限）
- class label: seedごとのランダム選択、class固定指定なし
- generation/FID batch: `64`、GPU数: `1`
- FID reference: `fid-refs/cifar10-32x32.npz`

2 checkpoint × 2条件 = **4 FID評価、20万画像**。
主解析はcheckpointごとの`FID(ODE) - FID(Restart)`で、正値がRestartによる改善を示す。
1範囲の評価なので、seed間変動や統計的有意性はこの実験だけでは推定しない。

## 所要時間と保存容量

既存実測とNFE比例による概算。GPU待機、初回モデル・Inception detectorの取得を除く。

| GPU | 1条件 | 全4条件 | 運用上の確保時間 |
| --- | ---: | ---: | ---: |
| Quadro RTX 8000 | 約1.4時間 | **約5.6時間** | 6〜8時間 |
| NVIDIA TITAN X (Pascal) | 約2.3時間 | 約9.3時間 | 10〜12時間 |

RTX 8000は実験8-3の18-step ODE生成平均`4243秒/50k`を`39/35`倍し、FID平均
`294秒`を加算する。全4条件は`4*(4243*39/35+294)/3600 ≈ 5.6時間`。
TITAN Xは実験8-4のFID-25kで、既存5k画像に追加20kを生成したK=1の平均`3309秒`と
FID平均`66秒`に基づき、`4*(3309*50/20+66*2)/3600 ≈ 9.3時間`とする。
FID時間の比例換算とODE/Restartの同NFE換算はいずれも概算。

生成画像は約0.8〜1.2 GiB、checkpointとInception detectorのcache容量は別途必要。
実測の生成・FID時間は結果TSVに保存する。生成画像、log、ダウンロードしたmodelはcommitしない。

## 計画確認と検証

以下はリポジトリ直下から実行する。計画表示はGPU待機、モデル取得、ファイル書込みを行わない。

```bash
PLAN_ONLY=1 \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/schedule_after_gpu_free.sh

experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/schedule_after_gpu_free.sh --help

/home/kyoukinkin/data/conda-envs/edm/bin/python -m compileall \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts

/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/test_experiment.py
```

実GPUが利用できる環境で、両checkpoint・両条件について固定seed`0-1`のsmoke testを行う。
FID-50k画像とは別のdirectoryに保存する。`SAMPLES_ONLY=1`もGPU待機対象。

```bash
SAMPLES_ONLY=1 SAMPLE_SEEDS=0-1 GPU_INDEX=0 \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

## 実行と部分実行

```bash
tmux new -s edm-exp8-7
# tmux内でリポジトリ直下に移動して実行する。
GPU_INDEX=0 EXPECTED_GPU_NAME='Quadro RTX 8000' \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

detachは`Ctrl-b`の後に`d`。再接続は`tmux attach -t edm-exp8-7`。
TITAN Xを使う場合は、そのphysical indexと`EXPECTED_GPU_NAME='NVIDIA TITAN X (Pascal)'`を指定する。
`CHECK_INTERVAL_SEC`の既定値は300秒。共通wrapperが`CUDA_VISIBLE_DEVICES`をphysical indexに設定する。
GPUがジョブスケジューラから排他的に割当済みの場合に限り、`run_restart_fid.sh`を直接起動できる。

checkpointを限定する場合:

```bash
CHECKPOINT_FILTER='exp3-best' GPU_INDEX=0 \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

条件を限定する場合:

```bash
CONDITION_FILTER='pretrained-restart-k1 pretrained-ode-nfe39' GPU_INDEX=0 \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/schedule_after_gpu_free.sh
```

filterは空白区切り。存在しないcheckpoint/conditionはエラー。
片方だけの条件も実行できるが、比較値とグラフは両条件が揃ってから出力する。

`PYTHON`、`TORCHRUN`で実行環境、`EXP_ROOT`で保存先、`CONFIG`で条件TSV、`FID_REF`で
FID referenceを変更できる。既定の条件TSVはscript側の固定pathにあるため、保存先だけを
変更しても利用できる。`GEN_BATCH`、`FID_BATCH`も変更可能（既定64）。
seed範囲、画像数、rho、noise、Kとstepsは本実験の設定に固定する。

modelとInception detectorはdnnlibのURL cacheを再利用する。既定cacheは
`<EXP_ROOT>/.cache/dnnlib/downloads/`。既存cacheを使うには`DNNLIB_CACHE_DIR`を指定する。

## 監視・停止・再開

```bash
tail -f experiments/8/7_trained_restart_k1_fid50k_1seed/logs/schedule-after-gpu-free.log
```

wrapperのlogにはrunnerの進捗は含まれない。tmux出力、またはcondition別の
`logs/<condition>-generate.log`と`logs/<condition>-fid.log`を確認する。

現在の1条件が終わったところで停止する:

```bash
touch experiments/8/7_trained_restart_k1_fid50k_1seed/STOP_AFTER_CURRENT_CONDITION
```

再開前に停止fileを削除し、同じ実行コマンドを使う。起動時に停止fileがあれば条件を開始しない。
FID記録済みならskip、未記録でも50,000枚の期待seed画像が揃っていれば生成をskipしてFIDを計算する。
生成中に中断した場合は同じseed範囲を再生成し、同じ画像pathへ保存する。
画像数だけでなく、欠損・重複path・余分なseedを検証してからFIDへ進む。

画像directoryの`.protocol.json`と結果TSVの`protocol_id`で設定変更を検出する。
checkpoint・referenceのlocal file情報、sampler設定、batch、生成/FIDコードが変わった場合、
既存結果や画像を別条件として流用しないようエラーにする。その場合は新しい`EXP_ROOT`を指定する。

## 結果

```text
images/<condition>/fid-n50000-seeds-0-49999/
images/<condition>/samples-seeds-0-1/
logs/<condition>-{generate,fid,samples}.log
results/restart_fid_summary.tsv
results/restart_improvements.tsv
results/restart_k1_fid50k_overview.{png,svg}
```

- `restart_fid_summary.tsv`: 各条件のFID、checkpoint、sampler、seed、reference、batch、所要時間。
- `restart_improvements.tsv`: checkpointごとのFIDと改善幅。片方が未完了なら`complete=0`、改善幅は`NA`。
- グラフ: 完了したpairのFIDと改善幅。単一評価なのでerror barは付けない。

runnerは各評価の記録後と終了時に集計・描画する。手動再生成はGPU待機対象外:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/summarize_results.py
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/plot_results.py
```
