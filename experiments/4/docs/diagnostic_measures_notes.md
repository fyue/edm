# Prediction Inconsistency と ODE 更新方向の診断メモ

## 背景

EDM CIFAR-10 class-conditional の `S_churn` 感度実験では、低kimg checkpointほど `S_churn` によるFID改善が大きく、pretrained modelでは効果がほぼ見られなかった。

この傾向を説明するために、単にFIDを見るだけでなく、サンプリング中のモデル予測の不安定性を診断量として測り、FIDや `S_churn` 改善幅との相関を見る。

## 1. Prediction inconsistency の意味

EDMでは、モデル出力 `D(x, sigma)` は基本的にscoreそのものではなく、clean image `x0` の推定、つまり denoised prediction である。

理想的には、近い入力や近い `sigma` に対して、`D(x, sigma)` も一貫した出力になるはずである。

例えば、次のような量を測る。

```text
I_pred = || D(x_t, sigma_t) - D(x_s, sigma_s) ||
```

ここで `x_t` と `x_s`、または `sigma_t` と `sigma_s` が近いにもかかわらず `I_pred` が大きい場合、denoised prediction が局所的に不安定である可能性がある。

## 2. 1 full step vs 2 half steps の不一致

同じ始点から、1回の大きなstepで進めた結果と、2回の半分stepで進めた結果を比較する。

```text
1 full step:
x_t -> x_next_full

2 half steps:
x_t -> x_mid -> x_next_half
```

診断量は例えば次のように定義できる。

```text
I_step = || x_next_full - x_next_half ||
```

`I_step` が大きい場合、その区間ではODE更新が不安定、またはstep幅に対してモデル予測が十分に一貫していない可能性がある。

## 3. EDMにおけるDとscore場・ODE driftの関係

注意すべき点として、EDMの `D(x, sigma)` はscore場そのものではない。

ただし、EDM samplerでは `D(x, sigma)` から更新方向が計算される。簡略化すると、

```text
d = (x - D(x, sigma)) / sigma
```

のような方向がODEの更新に使われる。

したがって、正確には「Dの予測場が滑らか」と言うより、

```text
denoised prediction D(x, sigma) が一貫している
-> Dから誘導されるODE更新方向 d が安定している
-> samplerの軌道が破綻しにくい
```

と表現するのがよい。

## 4. Pretrained model と未成熟checkpointの理想的な違い

未成熟checkpointでは、`D(x, sigma)` の推定誤差が大きく、近い入力や近い `sigma` でも出力がぶれやすいと考えられる。

その結果、

```text
D(x, sigma) が不安定
-> d = (x - D(x, sigma)) / sigma も不安定
-> 1 full step と 2 half steps の結果がずれやすい
-> S_churn やstep配置の影響が大きく出る
```

一方で pretrained model では、`D(x, sigma)` がより正確で一貫しているため、誘導されるODE更新方向も安定しやすい。

そのため、大きめのstepでも破綻しにくく、`S_churn` のような確率的摂動を加えても改善幅は小さい、または悪化する可能性がある。

## 5. 相関として見るべきもの

各checkpointについて、次の関係を見る。

```text
診断量 I_pred または I_step
vs
FID
```

または、

```text
診断量 I_pred または I_step
vs
FID(S_churn=0) - FID(best S_churn)
```

もし低kimg checkpointほど診断量が大きく、同時に `S_churn` 改善幅も大きければ、

```text
未成熟モデルでは、denoised predictionの不安定性やODE更新方向の不安定性が大きく、
そのためsampling hyperparameterへの感度が高い
```

という説明につながる。

## 6. 使いやすい表現

避けたい表現:

- `D` の予測場が滑らか
- EDMのscore場が滑らか

より正確な表現:

- denoised prediction `D(x, sigma)` の一貫性
- `D(x, sigma)` から誘導されるODE更新方向の安定性
- ODE drift の局所的な不安定性
- 近い入力に対するdenoiser出力の変動
- 1 full step と 2 half steps の不一致
