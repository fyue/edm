# Experiment 5-13 trajectory-distance results

## Condition

- seeds: `0-255`
- steps: `18`
- sigma range: `80.0` to `0.002` with `rho=7.0`
- RK4 subdivisions: `4` and `8` per original interval
- estimated NFE per image: `854`
- measured elapsed time: `8.73` minutes

## Final unquantized output distance

| comparison | mean | median | std | p90 |
|---|---:|---:|---:|---:|
| RK4-4 vs RK4-8 | 1.39246818066e-05 | 5.07169588638e-06 | 2.9151959837e-05 | 2.55165341239e-05 |
| 5-4 variable AB vs reference | 0.0741361291537 | 0.0684446944248 | 0.0350531912404 | 0.105940687414 |
| 5-11 fixed AB vs reference | 0.0262701012719 | 0.0168729256317 | 0.0317229774598 | 0.0496878439588 |

The reference final output is `D(x_ref(0.002), 0.002)`; the ODE is never evaluated at sigma=0.

## Outputs

- `per_seed_metrics.tsv`: scalar relative-L2 values for every seed and point
- `step_stats.tsv`: mean, median, sample standard deviation, and 90th percentile
- `trajectory_distances.svg`: log-sigma plots; sigma=0 final values are listed above
