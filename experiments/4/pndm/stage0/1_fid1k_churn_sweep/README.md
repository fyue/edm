# PNDM Stage 0: FID-1k S_churn sweep

## 目的

4つの学習段階について `S_churn=0-7` を低コストに評価し、PNDM samplerの
checkpoint依存性とchurn感度を確認する。

## 条件

- sampler: `pndm`
- checkpoints: `10k, 40k, 120k, pretrained`
- `S_churn`: `0,1,2,3,4,5,6,7`
- samples: `1024`
- seeds: `0-1023`
- steps: `18`
- 条件数: `32`

## 実行

```bash
PLAN_ONLY=1 experiments/4/pndm/stage0/1_fid1k_churn_sweep/scripts/run.sh
experiments/4/pndm/stage0/1_fid1k_churn_sweep/scripts/run.sh
```

一部だけ実行する場合:

```bash
CHECKPOINT_FILTER="10k pretrained" CHURN_FILTER="0 3 7" \
  experiments/4/pndm/stage0/1_fid1k_churn_sweep/scripts/run.sh
```

## 描画

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/4/pndm/scripts/plot_fid_churn.py \
  --summary experiments/4/pndm/stage0/1_fid1k_churn_sweep/results/fid_summary.tsv \
  --out experiments/4/pndm/stage0/1_fid1k_churn_sweep/results/fid1k_churn_curve.png \
  --title "PNDM Stage 0: FID-1k S_churn sweep"
```

FID-1kは正式値ではなく、同じ条件内の相対比較にのみ使用する。
