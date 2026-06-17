# EDM CIFAR-10 本番学習メモ

## 目的

CIFAR-10 class-conditional EDM/DDPM++ を、論文の **FID 1.79** にできるだけ近い条件で再現する。

論文との差分は、計算環境が論文の `8xV100` ではなく、こちらは **Quadro RTX 8000 1基** であること。

## 環境

`edm` conda環境を使用する。GPUは常に以下で固定する。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
```

確認済みのPyTorch環境:

```text
torch 1.12.1+cu113
cuda available True
device Quadro RTX 8000
```

PyTorch 1.12.1 と互換にするため、NumPyは `1.26.4` に調整済み。

## データ

CIFAR-10 dataset作成済み。

```text
datasets/cifar10-32x32.zip
fid-refs/cifar10-32x32.npz
```

ただし、論文/公式pretrained比較では、公式FID referenceを使う。

```text
https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

## 本番設定

READMEのCIFAR-10 class-conditional VP設定に対応する。

```bash
--cond=1
--arch=ddpmpp
--precond=edm
--duration=200
--batch=512
--lr=0.001
--ema=0.5
--dropout=0.13
--augment=0.12
--fp16=False
```

1 GPU運用のため、以下を使う。

```bash
--batch-gpu=128
```

これは有効バッチサイズ `--batch=512` を変えず、勾配蓄積の分割だけを変える設定。論文設定からのズレは小さい。

## batch-gpu比較

短時間ベンチ済み。

```text
batch-gpu=64:  約6.6〜6.7 sec/kimg, gpumem 8.89GB
batch-gpu=128: 約6.4〜6.5 sec/kimg, gpumem 16.50GB
batch-gpu=256: 約6.35〜6.42 sec/kimg, gpumem 31.74GB
```

`256` が少し速いがメモリ余裕が小さいため、長期運用では `128` を採用する。

## 想定時間

学習本体:

```text
200,000 kimg × 約6.4 sec/kimg
= 約15日
```

評価は `3k, 10k, 20k, 40k, 80k, 120k, 160k, 200k` で入れる。
途中評価は1 seed範囲だけ、最終200kだけ3 seed範囲で正式評価する。

FID評価回数は合計10回相当。

```text
途中8点: 0-49999
最終追加: 50000-99999, 100000-149999
```

総合見込み:

```text
約15〜16日
```

早い段階の `3k` と `10k` で、学習が破綻していないかをかなり早く確認できる。

## 途中確認方針

指定した評価点 `3k, 10k, 20k, 40k, 80k, 120k, 160k, 200k` ごとに以下を行う。

```text
1. 学習を一旦正常終了
2. health.tsv に状態記録
3. 少数sample画像生成
4. FID用50,000枚生成
5. FID計算
6. --resume で次の評価点へ
```

FID評価は公式referenceを使用する。途中評価は保存済みsnapshotを読むだけで、学習重みは変更しない。

## 管理スクリプト

本番学習は以下のスクリプトで管理する。

```text
scripts/cifar10_cond_train_eval_resume.sh
```

主な機能:

```text
tmux前提
3k, 10k, 20k, 40k, 80k, 120k, 160k, 200kで学習/FID評価
途中評価は1 seed範囲
最終200kだけ3 seed範囲で正式評価
training-stateから自動再開
既に記録済みFIDはスキップ
health.tsv記録
sample画像生成
seed固定
STOP_AFTER_CURRENT_STAGEで安全停止
```

`nohup` は `torchrun` がSIGHUPで終了したため非推奨。`tmux` を使う。

## 起動方法

```bash
tmux new -s edm
scripts/cifar10_cond_train_eval_resume.sh
```

tmuxから切り離す。

```text
Ctrl-b -> d
```

tmuxに戻る。

```bash
tmux attach -t edm
```

## 確認コマンド

進捗:

```bash
tail -f logs/cifar10-cond-train-eval-resume.log
```

FID結果:

```bash
cat fid-evals/cifar10-cond-ddpmpp-edm-managed/summary.tsv
```

健全性:

```bash
cat fid-evals/cifar10-cond-ddpmpp-edm-managed/health.tsv
```

安全停止:

```bash
touch training-runs/cifar10-cond-ddpmpp-edm-managed/STOP_AFTER_CURRENT_STAGE
```

## 事前学習済みモデルでの確認

公式pretrainedで3 seed範囲を評価済み。

```text
seeds 0-49999:        FID 1.8478
seeds 50000-99999:    FID 1.82278
seeds 100000-149999:  FID 1.84661
best:                 FID 1.82278
```

論文値 `1.79` との差は `+0.03278`。かなり近く、評価パイプラインは正しく動いていると見てよい。
