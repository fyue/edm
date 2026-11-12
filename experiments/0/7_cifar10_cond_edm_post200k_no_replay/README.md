# 実験0-7：乱数再生を避けた200k→300k追加学習

0-1の200kから未使用seedで300kまで継続学習し、後続のRestart評価に使う10k刻みのcheckpointを作る。起動後はGPU待機・学習中として扱い、300k完了は検証結果で確認する。

## 条件と対象

[issue #2](https://github.com/fyue/edm/issues/2)の学習・checkpoint検証のみを行う。FID評価・画像生成・Restart走査は行わない。
0-1の200k stateからnet・Adam、対応snapshotからEMAを復元し、0-1と0-3の既存成果物は変更しない。乱数再生が確認された0-3は起点にしない。

| 項目 | 値 |
|---|---|
| データ | CIFAR-10全50,000枚、`datasets/cifar10-32x32.zip` |
| モデル・損失 | class-conditional DDPM++ / EDM |
| optimizer / LR | Adam / 0.001、ramp-upは累積学習量に従う |
| EMA | 0.5 Mimg |
| batch / batch-gpu / GPU数 | 512 / 128 / 1 |
| 精度 / dropout / augment | FP32 / 0.13 / 0.12 |
| 起点 / 終点 | 200000 / 300000 kimg（`--duration=300`） |
| 初回seed | 200000（親のseed=0は再使用しない） |
| 保存点 | 210k, 220k, …, 300kの10組、`config/checkpoint_kimgs.txt` |
| 通常tick保存 | `--snap=0 --dump=0`で無効 |

指定学習量に到達する最初のbatch直後に保存するため、実進捗とラベルには0.512 kimg未満の差がある。再開時は既存CLIの仕様どおりラベルから累積進捗を復元する。
追加100kは既存実績で約7.5日、checkpoint保存は約8.3 GiBを見込む。

## 実行と確認

リポジトリ直下で実行する。`PYTHON`にはedm環境のPythonを指定できる。

```bash
RUN=experiments/0/7_cifar10_cond_edm_post200k_no_replay/scripts/run.sh
bash "$RUN" --help
bash "$RUN" --plan-only
bash "$RUN" --dry-run
bash "$RUN"
bash "$RUN" --verify-only
```

`run.sh`および`schedule_after_gpu_free.sh`は共通GPU待機スクリプトを呼ぶ。ローカルGPU設定を読み、GPU名を照合し、設定GPUが使用中なら待機する。他のGPUへ切り替えない。help・plan・dry-run・verifyではGPU設定読み込み・待機を行わない。

このPCでの永続セッション名は`edm-exp0-7`。ログは`logs/master.log`、待機ログは`logs/schedule-after-gpu-free.log`。`tmux attach -t edm-exp0-7`で確認できる。
学習区間200k→300kは原則1回の起動で実行し、途中FID評価のために停止しない。

## 再開とseed予約

中断後は同じコマンドを実行する。snapshot/stateペアを新しい順にCPUで読み、EMA・net・optimizerの有限値と構造を確認して、最新正常ペアを選ぶ。片側欠損・空・破損ペアは再開に使わず、理由を`results/resume_validation.json`へ記録する。再保存で衝突する不完全ファイルは`training-runs/quarantine/`へ退避して保持する。正常な保存点がなければ0-1の200kへ戻る。

再開seedは再開kimg以上で履歴にない最小値を使う。同じ200kからの再試行なら200001、200002と進む。親seed=0も除外する。
runnerの排他ロック下で実起動前に履歴をfsyncして予約するため、起動失敗・強制終了でもseedを再使用しない。履歴が破損している場合は停止し、空の履歴として扱わない。plan・dry-runはseedを消費しない。

`results/launch_history.jsonl`はstart/endの追記形式で、日時（タイムゾーン付き）、seed、再開元、開始・終了進捗、終了状態を記録する。強制終了でendがないstartも使用済みseedとして扱う。
最大で直近10k分を再実行する。欠損した過去checkpointがある場合は全10点の完了検証を通さない。

未使用seed方式は同じ乱数列・データ順の先頭からの再生を避けるための措置であり、0→300k無停止学習との完全一致は保証しない。個々の画像は通常の学習でも再出現する。厳密なRNG・データ順保存復元は対象外。

## 成果物・完了条件

- `training-runs/cifar10-cond-ddpmpp-edm-post200k-no-replay/`：checkpointと学習ログ（Git対象外）。
- `results/experiment_manifest.json`：学習条件とコード・データ・起点checkpointのSHA256。再開時の不一致を拒否する。
- `results/runtime.json`：Python・PyTorch・CUDAとGPU UUID・driver。
- `results/launch_history.jsonl`：全起動の追記履歴。
- `results/checkpoint_manifest.tsv`、`results/health.json`：全10点のCPU読込検証と到達進捗・loss。

300k到達、記録lossの有限値、全10ペアの読込成功、一覧・履歴保存で完了とする。学習終了後に自動検証し、成功した場合だけ本README・グループREADME・監査の状態を完了へ更新し、INDEXを再生成する。途中中断を完了として扱わない。

```bash
python -m compileall -q experiments/0/7_cifar10_cond_edm_post200k_no_replay/scripts
python experiments/0/7_cifar10_cond_edm_post200k_no_replay/scripts/test_experiment.py -v
python train.py --help
bash "$RUN" --dry-run
python scripts/build_experiment_index.py --check
```
