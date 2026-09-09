# PNDM Stage 1: FID-5k S_churn sweep

## 目的

Stage 0より広いcheckpoint集合で `S_churn=0-7` を評価し、FID-50kへ進める
代表条件を選択する。

## 条件

- sampler: `pndm`
- checkpoints: `5.5k, 10k, 15.0k, 20k, 40k, 80k, 120k, pretrained`
- `S_churn`: `0,1,2,3,4,5,6,7`
- samples: `5000`
- seeds: `0-4999`
- steps: `18`
- 条件数: `64`

## 実行

```bash
PLAN_ONLY=1 experiments/4/pndm/stage1/1_fid5k_churn_sweep/scripts/run.sh
experiments/4/pndm/stage1/1_fid5k_churn_sweep/scripts/run.sh
```

## 描画

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/4/pndm/scripts/plot_fid_churn.py \
  --summary experiments/4/pndm/stage1/1_fid5k_churn_sweep/results/fid_summary.tsv \
  --out experiments/4/pndm/stage1/1_fid5k_churn_sweep/results/fid5k_churn_curve.png \
  --title "PNDM Stage 1: FID-5k S_churn sweep"
```

Stage 1終了後、checkpointごとの有望条件を確認し、Stage 2の
`experiments/4/pndm/stage2/1_fid50k_selected_churn/config/selected_conditions.tsv`
と `experiments/4/pndm/stage2/config/paired_conditions.tsv` を確定する。
