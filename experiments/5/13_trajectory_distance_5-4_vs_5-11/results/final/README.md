# Experiment 5-13 trajectory-distance results

## Condition

- seeds: `0-1023`
- steps: `18`
- sigma range: `80.0` to `0.002` with `rho=7.0`
- RK4 subdivisions: `4` and `8` per original interval
- estimated NFE per image: `854`
- measured elapsed time: `34.86` minutes

## Final unquantized output distance

| comparison | mean | median | std | p90 |
|---|---:|---:|---:|---:|
| RK4-4 vs RK4-8 | 1.19301230093e-05 | 4.98916576602e-06 | 2.99186504368e-05 | 2.3282193358e-05 |
| 5-4 variable AB vs reference | 0.0719538746091 | 0.066418929496 | 0.0322693493806 | 0.108227169393 |
| 5-11 fixed AB vs reference | 0.0228559706225 | 0.015614965095 | 0.0229992358307 | 0.043154325692 |

The reference final output is `D(x_ref(0.002), 0.002)`; the ODE is never evaluated at sigma=0.

## Outputs

- `per_seed_metrics.tsv`: scalar relative-L2 values for every seed and point
- `step_stats.tsv`: mean, median, sample standard deviation, and 90th percentile
- `trajectory_distances.svg`: log-sigma plots; sigma=0 final values are listed above
