# Experiment 7-3: FM Solver/NFE Sweep and EDM Comparison

20k学習のFMとEDMで、solverとNFEを変えてFID-5kを比べる。FMはNFEとともに改善し、Heunは5・9 NFEでわずかに良く、19・35 NFEでは逆。18 NFEのEulerではFM 11.79がEDM 12.28より良い。

## 目的

Experiment 7-2で20,000 kimgまで学習したlinear FMについて、生成品質の差が
学習誤差ではなく数値積分誤差から来ている可能性を調べる。同じFM velocity場を
Euler/Heunで積分し、同一NFEでFID-5kを比較する。さらに、学習条件と学習進行度が
同じ20k EDM checkpointを使い、18 NFEと35 NFEでFM対EDMを比較する。

この実験は学習を行わず、保存済みcheckpointの評価だけを行う。

## 比較設計

### FM内のsolver比較

| NFE | Euler steps | Heun steps | 目的 |
|---:|---:|---:|:---|
| 5 | 5 | 3 | 極低NFEでのsolver差 |
| 9 | 9 | 5 | 低NFEでのsolver差 |
| 19 | 19 | 10 | pilot既定付近のsolver差 |
| 35 | 35 | 18 | EDM既定と同じ計算量 |

FM Heunは最後の区間のみEulerなので`NFE = 2 × steps - 1`、FM Eulerは
`NFE = steps`となる。

### FM対EDM

| NFE | FM | EDM | 揃えるもの |
|---:|:---|:---|:---|
| 18 | FM Euler 18 steps | EDM Euler 18 steps | 1次solver・NFE |
| 35 | FM Heun 18 steps | EDM Heun 18 steps | 2次solver・steps・NFE |

EDM Eulerは標準EDMのrho discretization、linear sigma schedule、scalingなしを
維持し、solverだけをEulerへ変更する。両モデルともCIFAR-10 conditional、DDPM++、
batch 512、LR 0.001、EMA 0.5、dropout 0.13、augment 0.12、FP32、seed 0、20kである。

全条件をseeds 0-4999のFID-5kで評価し、目視用にseeds 0-63も保存する。条件一覧と
checkpointは[config/conditions.tsv](config/conditions.tsv)に固定している。

## 仮説と読み方

- Heunが同一NFEのEulerより良い場合、FMの誤差には数値積分誤差が大きく寄与する。
- NFEを増やしてもFIDがほぼ改善しない場合、20k時点では学習誤差が支配的である。
- 同一solver・同一NFEでもEDMが良い場合、単なるsolver差ではなく学習目的または
  probability pathの差が残る。
- FID-5kには標本変動があるため、小差を最終結論にせず、有望条件だけFID-50kで
  再評価する。

既存値はFM Euler 18 NFEが11.7926、EDM Heun 35 NFEが7.6149。ただし7-3では
同じ結果表へまとめるため、既存の5,000枚を再利用してFIDを再計算する。

## 実行

GPUを使わずに全11条件と入力ファイルを確認する。

```bash
PLAN_ONLY=1 \
  experiments/7/3_fm_solver_nfe_and_edm_comparison/scripts/schedule_after_gpu_free.sh
```

GPUが空くまで待ってから評価を開始する。

```bash
tmux new -s edm-exp7-3
  experiments/7/3_fm_solver_nfe_and_edm_comparison/scripts/schedule_after_gpu_free.sh
```

特定条件だけを実行する場合は空白区切りで指定する。

```bash
CONDITION_FILTER="fm-euler-nfe18 edm-euler-nfe18" \
  experiments/7/3_fm_solver_nfe_and_edm_comparison/scripts/schedule_after_gpu_free.sh
```

現在の条件終了後に安全停止するには次を作る。

```bash
touch experiments/7/3_fm_solver_nfe_and_edm_comparison/STOP_AFTER_CURRENT_CONDITION
```

再実行すると結果TSVに記録済みの条件をスキップする。生成が途中で止まった条件は、
同じseed範囲を再生成してからFID計算を続行する。

## 出力

```text
experiments/7/3_fm_solver_nfe_and_edm_comparison/
  images/<condition>/samples-seeds-0-63/
  images/<condition>/fid-n5000-seeds-0-4999/
  logs/fm-solver-nfe-comparison.log
  results/fm_solver_nfe_comparison.tsv
```

結果TSVにはFIDだけでなくmodel、sampler、solver、steps、NFE、seed範囲、生成時間、
FID計算時間、checkpoint、画像directoryを記録する。既存画像を再利用した条件は
`image_source=reused`、`generate_sec=NA`になる。

新規生成は9条件×5,000枚。7-2の実測から、固定seed画像とFID計算を含めて約50〜70分、
追加ディスク使用量は約180〜220 MBを見込む。

## 結果のグラフ

結果TSVから、FM内のNFE/solver比較と、同一solver・同一NFEでのFM対EDM比較を
1枚の図にまとめる。GPUは不要で、PNGと論文貼り込み用PDFを同時に生成する。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/7/3_fm_solver_nfe_and_edm_comparison/scripts/plot_results.py
```

```text
results/fid5k_solver_nfe_overview.png
results/fid5k_solver_nfe_overview.pdf
```

FID-5kはスクリーニング指標なので、特に0.1程度の小差をsolver間の有意な差として
解釈しない。FM対EDMの最終的な主張には、代表条件のFID-50kを使用する。
