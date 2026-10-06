# Experiment 8-4: FID by checkpoint and Restart K

FID-25,000 (lower is better). The minimum FID in each row is bold.

| Checkpoint | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | Best K | Best FID | 2nd K | Gap | Gain vs K=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80k | 2.48044 | 2.42064 | 2.43418 | **2.41438** | 2.42379 | 2.45124 | 2.46926 | 2.46158 | 2.47622 | 3 | 2.41438 | 1 | 0.00626 | 0.06606 |
| 90k | 2.49582 | **2.41795** | 2.44369 | 2.43590 | 2.42265 | 2.45334 | 2.47344 | 2.48622 | 2.49207 | 1 | 2.41795 | 4 | 0.00470 | 0.07787 |

Gap = second-best FID - best FID. Gain vs K=0 = FID(K=0) - best FID.
