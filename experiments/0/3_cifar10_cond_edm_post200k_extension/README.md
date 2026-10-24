# 実験0-3：200k以降の継続学習

実験0-1の200kから学習率0.001を維持して210kまで継続し、FIDの改善を調べる。
元の実験0-1を読み取り専用の出発点として使い、この実験のtraining-runsへ保存する。
今回は210kで停止する。220k以降への延長は明示的にtargetを指定する。

## 条件

CIFAR-10全50,000枚、class-conditional DDPM++/EDM、Adam lr=0.001、EMA=0.5 Mimg、
batch=512、batch-gpu=128、FP32、dropout=0.13、augment=0.12、学習seed=0、1 GPU。
200kのnetとoptimizerはtraining-stateから、EMAは対応snapshotから復元する。
乱数列・dataset iteratorは再初期化され、0→210kの無停止学習とはbitwise一致しない。
学習率のramp-upは累積学習量に従うため、200kで再度ramp-upしない。

202.5k、205k、207.5k、210kでsnapshot/stateを保存し、区間内は同じプロセスで学習する。
batch=512のため、指定画像数に到達する最初のupdateで保存する。checkpointラベルと実際の
進捗には0.512 kimg未満の差があり、再開時は既存仕様に従いラベルから進捗を復元する。
1ペア約0.83 GiB、4ペア約3.33 GiB。従来GPU実績では追加10kに約18時間を要する。

## 実行

リポジトリ直下でedm環境をactivateする。別の環境からは`PYTHON`でedmのPythonを指定する。
torchrunは同じPythonのbinを既定とし、必要なら`TORCHRUN`で指定できる。

```bash
RUN=experiments/0/3_cifar10_cond_edm_post200k_extension/scripts/run.sh
bash "$RUN" --help
bash "$RUN" --plan-only

# tmuxで学習→一次評価→条件成立時だけ追加評価
tmux new-session -s edm-exp0-3
bash experiments/0/3_cifar10_cond_edm_post200k_extension/scripts/schedule_after_gpu_free.sh

# 別ターミナルで確認
tail -f experiments/0/3_cifar10_cond_edm_post200k_extension/logs/master.log
```

CPU coordinatorが各GPU stageの前に共通`scripts/wait_for_gpu_and_run.sh`を呼ぶ。
`~/.config/edm/gpu.env`（または`EDM_GPU_CONFIG`）の`GPU_INDEX`と`EXPECTED_GPU_NAME`が必要。
共有コードにGPU番号・名前を固定せず、指定GPUが使用中なら待機し、別GPUへ切り替えない。
GPU設定・driver問い合わせが失敗した場合は学習・生成を開始しない。
help、plan、verifyはGPU設定読み込み・待機を行わない。

## FID比較

200kと210kを同じ物理GPU、driver、CUDA/PyTorch環境で新規評価する。
既存200kのFID値1.80679（K=0）、1.79334（K=1）は参考値であり、判定の基準には再利用しない。

- 通常EDM：Heun、main=18、K=0、NFE=35。
- Restart：main=18、K=1、restart steps=3、t_min=0.14、t_max=0.30、NFE=39。
- 共通：rho=7、sigma_min=0.002、sigma_max=80、S_churn=0、restart_noise=1。
- FID-50k、生成/FID batch=64、参照`fid-refs/cifar10-32x32.npz`。ラベルもseedに対応する乱数で決定する。

最初はseed `0-49999`で両checkpoint×両samplerの4評価を行う。
新規評価の`FID(200k,K=0) - FID(210k,K=0) >= 0.01`なら、両checkpoint・両samplerを
`50000-99999`と`100000-149999`でも評価する。一次4評価20万枚、全12評価60万枚。
既定`--fid-stage auto`。`screen`は一次のみ、`all`は判定にかかわらず3範囲を評価する。

平均、標本標準偏差（ddof=1）、paired差、改善した範囲数をresultsに保存する。
3範囲で平均FIDが低下し、2範囲以上で改善した場合を「改善傾向あり」とする。
通常EDMが主結果、Restartが副次結果。1範囲だけでは暫定値とし、未評価を悪化とは扱わない。
3範囲は固定モデルの生成seed変動を調べるもので、学習seed間変動や統計的有意差は評価しない。

## 停止・再開・延長

```bash
# 学習は次の2.5k checkpoint保存後、FIDは現在の1条件の保存後に終了
touch experiments/0/3_cifar10_cond_edm_post200k_extension/STOP_AFTER_CURRENT_STAGE

# STOPを削除し、同じコマンドで最新checkpointと不足画像から再開
unlink experiments/0/3_cifar10_cond_edm_post200k_extension/STOP_AFTER_CURRENT_STAGE
bash "$RUN"

# 学習のみ / 評価のみ / 検証のみ
bash "$RUN" --train-only
bash "$RUN" --eval-only --fid-stage all
bash "$RUN" --verify-only

# 明示的な次段階。210kのoptimizerとEMAから220kへ
bash "$RUN" --target-kimg 220000
```

STOPは自動削除しない。GPU待機中のSTOPはGPUが空いてworkerに移る時点でも確認する。
Ctrl-Cやプロセス強制終了後は最後の完成ペアまで戻るため、最大約2.5kを再実行する。
保存途中で片側欠損・空ファイルになった場合は安全のため拒否する。破損ファイルを保全して
対象ペアを出力先から退避させ、直前の正常なペアから再開する。

学習結果、コード・データ・モデルのSHA256、GPU UUID/driver/Python/PyTorch/CUDAを記録し、
再実行時の不一致は拒否する。変更した環境を使う場合は別実験として設定する。
220k以降は学習を延長し、評価対象を200kと指定targetにする。過去のFIDとbaselineは保持する。

## 成果物と検証

`results/experiment_manifest.json`、`runtime.json`、`health.json`、条件別FID JSON、
`comparison.tsv/.json`、`paired_improvements.tsv/.json`、`screening-<target>.json`を保存する。
FID JSONを完了マーカーとし、欠損画像のみ再生成する。再実行とverifyでは完了画像も検証する。
checkpoint、画像、ログはGit対象外。実行中にmanifest対象のコードを変更しない。

```bash
python -m compileall -q train.py training/training_loop.py experiments/0/3_cifar10_cond_edm_post200k_extension/scripts
python experiments/0/3_cifar10_cond_edm_post200k_extension/scripts/test_experiment.py -v
python train.py --help
bash "$RUN" --plan-only
```

完了条件は210kペア、有限loss、到達進捗、manifest、同一GPUの一次FID比較が揃うこと。
autoで閾値を満たした場合は3範囲の比較まで完了させる。
