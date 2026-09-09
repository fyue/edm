# PNDM Stage 2-2: extra seeds for representative pairs

## 目的

代表的な `S_churn=0` とselected churnのペアを追加2 seed範囲で評価し、
3 seed範囲のFIDとpaired improvementを集計する。

初期configとpaired conditionはEDM版の代表ペアに合わせている。PNDM Stage 2-1の
結果が異なる場合は、本実行前に両方を更新する。

pretrained、`S_churn=0` の追加seed結果はExperiment 5から再利用するため、
この実験ではselected churnだけを生成する。

## 実行

```bash
PLAN_ONLY=1 experiments/4/pndm/stage2/2_fid50k_extra_seeds/scripts/run.sh
experiments/4/pndm/stage2/2_fid50k_extra_seeds/scripts/run.sh
```

## 3 seed集計

```bash
experiments/4/pndm/stage2/2_fid50k_extra_seeds/scripts/summarize.sh
```

出力:

```text
results/fid50k_3seed_condition_stats.tsv
results/fid50k_3seed_improvement_stats.tsv
```
