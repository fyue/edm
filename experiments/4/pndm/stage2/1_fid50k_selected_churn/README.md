# PNDM Stage 2-1: selected S_churn FID-50k

## 目的

PNDM Stage 1で残った代表条件を50,000 samplesで評価する。初期configはEDM版の
FID-50k条件と対応しており、PNDM Stage 1の結果に応じて本実行前に更新する。

pretrainedの `S_churn=0` はExperiment 5の同一条件を再利用するため、実行configから
除外している。

## 実行

```bash
PLAN_ONLY=1 experiments/4/pndm/stage2/1_fid50k_selected_churn/scripts/run.sh
experiments/4/pndm/stage2/1_fid50k_selected_churn/scripts/run.sh
```

## 描画

既存のpretrained baselineも含めて描画する。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/4/pndm/scripts/plot_fid_churn.py \
  --summary experiments/4/pndm/stage2/1_fid50k_selected_churn/results/fid_summary.tsv \
  --summary experiments/4/pndm/stage2/config/reused_experiment5_pretrained.tsv \
  --seeds 0-49999 \
  --out experiments/4/pndm/stage2/1_fid50k_selected_churn/results/fid50k_churn_curve.png \
  --title "PNDM Stage 2: FID-50k S_churn sweep"
```
