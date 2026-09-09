# EDM CIFAR-10 における `S_churn` 感度解析

## 要旨

EDM CIFAR-10 class-conditional の学習途中 checkpoint に対して、sampling 時の `S_churn` が FID に与える影響を調べた。結論として、低 kimg の未成熟な checkpoint ほど `S_churn` による FID 改善が大きく、学習が進むにつれて改善幅は小さくなった。official pretrained model では改善はほぼ確認できず、seed ばらつきと同程度だった。

この結果は、未成熟なモデルほど sampler の確率的摂動や step 条件に敏感であり、十分訓練されたモデルではその感度が小さくなる可能性を示している。

## 目的

確認したい仮説は次の通りである。

1. 学習途中 checkpoint では、`S_churn` によって FID が改善する。
2. 改善幅は、学習が進んだ checkpoint ほど小さくなる。
3. pretrained model では、`S_churn` の効果はほぼノイズレベルになる。

## 実験の流れ

| stage     | 目的                       |                 評価画像数 | 主な条件                                         |
| --------- | -------------------------- | -------------------------: | ------------------------------------------------ |
| Stage 0   | 低コストなスモークテスト   |               1,024枚/条件 | `S_churn=0,5,20,40` など                       |
| Stage 1   | 広めのスクリーニング       |               5,000枚/条件 | 8 checkpoint x`S_churn=0-5`、後に `6,7` 追加 |
| Stage 2-1 | 代表条件の正式評価         |              50,000枚/条件 | Stage 1 から選んだ20条件                         |
| Stage 2-2 | seed 依存性の確認          | 50,000枚/条件 x 3 seed範囲 | `S_churn=0` と代表 best の paired 比較         |
| Stage 2-3 | `S_churn=6,7` の追加確認 |              50,000枚/条件 | 単一 seed 範囲で`0-7` を結合                   |

共通設定は CIFAR-10 32x32 class-conditional、EDM sampler、18 steps、`S_min=0, S_max=inf, S_noise=1` とした。

## Stage 0: FID-1k による予備確認

最初に、少数サンプルで `S_churn` を大きく変えたときの傾向を確認した。ここで `S_churn=20` と `40` が同じような結果になったため、EDM sampler の

```text
gamma = min(S_churn / num_steps, sqrt(2) - 1)
```

により、18 steps では `S_churn >= 7.46` 付近で churn 量が飽和することを確認した。そのため以降は、主に `S_churn=0-7` の範囲を調べた。

![Stage 0 fine S_churn sweep](../stage0/3_fine_s_churn_under_clip/results/stage0_fine_churn_curve.png)

## Stage 1: FID-5k によるスクリーニング

FID-5k で、複数 checkpoint に対して `S_churn=0-5` を評価した。低 kimg ほど `S_churn` による改善が大きく、pretrained ではほとんど変化がなかった。

![Stage 1 FID-5k S_churn sweep](fig1_stage1_fid5k_s_churn_sweep.png)

代表的な結果は次の通り。

| checkpoint | best`S_churn` | `S=0` FID-5k | best FID-5k | 改善幅 |
| ---------- | --------------: | -------------: | ----------: | -----: |
| 5.5k       |               5 |        14.8097 |     13.8606 | 0.9491 |
| 10k        |               2 |        10.2602 |      9.8452 | 0.4150 |
| 20k        |               3 |         7.6149 |      7.4403 | 0.1746 |
| 40k        |               3 |         6.9962 |      6.8641 | 0.1321 |
| 80k        |               3 |         6.7304 |      6.6451 | 0.0853 |
| 120k       |               3 |         6.7141 |      6.6658 | 0.0482 |
| pretrained |               3 |         6.5234 |      6.5149 | 0.0086 |

## Stage 2-1: FID-50k による代表条件の確認

Stage 1 で有望だった条件を FID-50k で再評価した。FID-5k で見えた傾向は、FID-50k でも概ね再現された。

![Stage 2-1 FID-50k S_churn sweep](fig2_stage2_exp1_fid50k_s_churn_sweep.png)

低 FID 側を拡大すると、80k、120k、pretrained の差は小さい。特に pretrained では `S_churn` による改善はほとんどない。

![Stage 2-1 low FID zoom](fig3_stage2_exp1_fid50k_low_fid_zoom.png)

| checkpoint | `S=0` FID-50k | best`S_churn` | best FID-50k | 改善幅 |
| ---------- | --------------: | --------------: | -----------: | -----: |
| 5.5k       |         10.5849 |               5 |       9.5387 | 1.0462 |
| 10k        |          5.7708 |               3 |       5.4177 | 0.3531 |
| 20k        |          2.9897 |               3 |       2.8271 | 0.1626 |
| 40k        |          2.3366 |               2 |       2.2152 | 0.1213 |
| 80k        |          2.0756 |               3 |       2.0054 | 0.0702 |
| 120k       |          2.0501 |               3 |       2.0137 | 0.0365 |
| pretrained |          1.8478 |               3 |       1.8432 | 0.0046 |

## Stage 2-2: 3 seed 範囲での paired improvement

単一 seed 範囲だけでは小さい差を過大解釈する可能性があるため、代表ペアについて追加の2 seed範囲を評価した。改善幅は同じ seed 範囲内で

```text
improvement = FID(S_churn=0) - FID(best S_churn)
```

として計算した。

![Stage 2-2 3-seed paired improvement](fig4_stage2_exp2_fid50k_3seed_paired_improvement.png)

| checkpoint |   `S=0` FID-50k | best`S_churn` |     best FID-50k |            改善幅 |
| ---------- | ----------------: | --------------: | ---------------: | ----------------: |
| 5.5k       | 10.4855 ± 0.0865 |               5 | 9.5926 ± 0.0470 |  0.8930 ± 0.1335 |
| 10k        |  5.7319 ± 0.0338 |               3 | 5.4406 ± 0.0226 |  0.2912 ± 0.0542 |
| 20k        |  2.9622 ± 0.0240 |               3 | 2.8424 ± 0.0176 |  0.1198 ± 0.0381 |
| 40k        |  2.3037 ± 0.0291 |               2 | 2.2387 ± 0.0223 |  0.0650 ± 0.0489 |
| 80k        |  2.0557 ± 0.0175 |               3 | 2.0118 ± 0.0074 |  0.0439 ± 0.0241 |
| 120k       |  2.0300 ± 0.0179 |               3 | 2.0044 ± 0.0084 |  0.0256 ± 0.0096 |
| pretrained |  1.8391 ± 0.0141 |               3 | 1.8416 ± 0.0017 | -0.0026 ± 0.0144 |

3 seed 平均でも、低 kimg ほど改善幅が大きい傾向は残った。一方で pretrained では平均改善幅が負になり、`S_churn` の効果は確認できなかった。

また、自前 checkpoint 内では `120k + S_churn=3` が平均 FID 2.0044 で最良だった。ただし `80k + S_churn=3` の 2.0118 との差は小さい。

## Stage 2-3: `S_churn=6,7` 追加結果

単一 seed 範囲の FID-50k では、`S_churn=6,7` も追加した。これは 3 seed 平均ではなく参考結果だが、低 kimg ではさらに大きい `S_churn` が最良になる場合があった。

![Stage 2-3 FID-50k S_churn 0-7](../stage2/3_fid50k_churn_6_7/results/fid50k_churn_0_7_curve.png)

| checkpoint | best`S_churn` | best FID-50k |
| ---------- | --------------: | -----------: |
| 5.5k       |               6 |       9.4993 |
| 10k        |               7 |       5.3391 |
| 20k        |               3 |       2.8271 |
| 40k        |               2 |       2.2152 |
| 80k        |               3 |       2.0054 |
| 120k       |               3 |       2.0137 |
| pretrained |               3 |       1.8432 |

この結果から、低 kimg では `S_churn=5` 付近で打ち切らず、clip 直前まで改善が続く可能性がある。ただし、この部分は単一 seed 範囲のため、主結論は Stage 2-2 の 3 seed 平均に基づける。

## 解釈

今回の結果は、`S_churn` の効果がモデルの学習段階に依存することを示している。

- 未成熟 checkpoint では、denoised prediction `D(x, sigma)` や、それに由来する ODE 更新方向が不安定で、sampling hyperparameter の影響が大きい可能性がある。
- 学習が進むと予測が安定し、`S_churn` による改善幅は小さくなる。
- pretrained model では `S_churn` の追加摂動は改善につながらず、むしろ seed によっては悪化する。

したがって、FID を比較するときは checkpoint の学習量だけでなく、sampling 設定も揃える必要がある。特に学習途中 checkpoint の比較では、`S_churn=0` 固定だと性能差の見え方が変わる可能性がある。

## まとめ

1. `S_churn` による FID 改善は、低 kimg checkpoint で大きい。
2. 改善幅は学習が進むほど小さくなる。
3. pretrained model では改善はほぼノイズレベルである。
4. 自前 checkpoint では、3 seed 平均で `120k + S_churn=3` が最良だった。
5. 低 kimg では `S_churn=6,7` まで改善が続く可能性があるが、これは追加 seed で確認する必要がある。

## 今後

- `5.5k, 10k` について `S_churn=6,7` の追加 seed を評価する。
- 生成画像を並べ、FID改善が見た目にも対応しているか確認する。
- prediction inconsistency や 1 full step vs 2 half steps の不一致を測り、`S_churn` 改善幅との相関を見る。
