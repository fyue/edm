# Experiment 6-1: EDM Forward-Noising Norm Density

## Purpose

PNDM論文 Figure 2 のように、前向きノイズ付加後のデータのノルム分布を可視化する。

PNDM/DDPMのVP型前向き過程は、

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
```

で全体スケールが保たれやすい。一方、EDMの訓練入力は、

```text
x_sigma = x_0 + sigma eps
```

なので、raw norm は `sigma` とともに大きくなる。そこで、この実験では次の2種類を作る。

- `scaled`: `||x_0 + sigma eps|| / sqrt(dim) / sqrt(sigma_data^2 + sigma^2)`
- `raw`: `||x_0 + sigma eps|| / sqrt(dim)`

`scaled` はPNDM Figure 2の「高密度帯を見る」用途に近く、`raw` はEDM入力そのもののスケール変化を見る用途に向く。

## Files

```text
experiments/6/1_edm_forward_norm_density/
  README.md
  scripts/
    plot_edm_forward_norm_density.py
    run_edm_forward_norm_density.sh
```

Outputs are written under:

```text
experiments/6/1_edm_forward_norm_density/results/
experiments/6/1_edm_forward_norm_density/logs/
```

## Default Settings

- dataset: `datasets/cifar10-32x32.zip`
- max images: `5000`
- sigma schedule: EDM rho schedule from `0` to `80`
- steps: `256`
- `rho=7`
- `sigma_data=0.5`

## Run

```bash
experiments/6/1_edm_forward_norm_density/scripts/run_edm_forward_norm_density.sh
```

This creates both:

```text
experiments/6/1_edm_forward_norm_density/results/edm_forward_norm_density_scaled.png
experiments/6/1_edm_forward_norm_density/results/edm_forward_norm_density_raw.png
```

## Small Smoke Run

```bash
MAX_IMAGES=1000 STEPS=128 \
  experiments/6/1_edm_forward_norm_density/scripts/run_edm_forward_norm_density.sh
```

## Single Mode

```bash
MODE=scaled \
  experiments/6/1_edm_forward_norm_density/scripts/run_edm_forward_norm_density.sh
```

or:

```bash
MODE=raw \
  experiments/6/1_edm_forward_norm_density/scripts/run_edm_forward_norm_density.sh
```

## Notes

This script does not use the neural network. It only uses the training dataset and EDM's forward noising form. GPU is not required.
