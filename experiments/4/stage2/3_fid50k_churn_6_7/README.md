# Experiment 4 Stage 2 Third: S_churn=6,7 FID-50k Extension

## 目的

Stage 2実験1を変更せず、同じcheckpoint・seed・sampler条件で
`S_churn=6,7` のFID-50kを追加評価し、既存結果と結合した結果画像を作る。

## 条件

- checkpoint: `5.5k, 10k, 20k, 40k, 80k, 120k, pretrained`
- `S_churn`: `6, 7`
- 新規条件数: 14
- samples: `50000`
- seeds: `0-49999`
- steps: `18`
- `S_min=0`, `S_max=inf`, `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`

`steps=18` では `gamma=min(S_churn/18, sqrt(2)-1)` のため、6と7は
どちらもclip前の異なるsampling条件になる。

## 実行

```bash
PLAN_ONLY=1 experiments/4/stage2/3_fid50k_churn_6_7/scripts/run_fid50k_churn_6_7.sh
experiments/4/stage2/3_fid50k_churn_6_7/scripts/run_fid50k_churn_6_7.sh
```

実行済み条件はsummaryを見てskipされる。現在の条件終了後に止める場合:

```bash
touch experiments/4/stage2/3_fid50k_churn_6_7/STOP_AFTER_CURRENT_CONDITION
```

## 出力

追加実験の生データ:

```text
images/<checkpoint>/S_churn-{6,7}/n50000-seeds-0-49999/
logs/fid50k-*.log
results/fid50k_selected_churn_summary.tsv
```

実験1と結合した成果物:

```text
results/fid50k_churn_0_7_combined.tsv
results/fid50k_churn_0_7_low_fid.tsv
results/fid50k_churn_0_7_best.tsv
results/fid50k_churn_0_7_curve.png
results/fid50k_churn_0_7_curve_low_fid.png
```

結合結果だけを再生成する場合:

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/4/stage2/3_fid50k_churn_6_7/scripts/merge_and_plot_fid50k_churn.py
```
