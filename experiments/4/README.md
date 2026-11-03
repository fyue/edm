# 実験4：S_churn感度（EDM sampler・PNDM）

## 問い

EDM samplerの確率的パラメータ`S_churn`の効果は、checkpointの学習段階によって変わるか。学習途中のモデルほどsampling設定の影響が大きいか。

## 主要な結論

- 学習初期のcheckpointほど`S_churn`による改善が大きく、学習が進むと改善幅は小さくなる。pretrainedでは改善がない。
- FID-50kの3 seed平均での改善幅（`S_churn=0` → 代表best）：5.5k 0.8930±0.1335（S=5）、10k 0.2912（S=3）、20k 0.1198（S=3）、40k 0.0650（S=2）、80k 0.0439（S=3）、120k 0.0256±0.0096（S=3）、pretrained −0.0026±0.0144（S=3）。
- 自前checkpointのうち平均で最良だったのは120k＋S_churn=3（2.0044±0.0084）。80k（2.0118）との差は小さい。
- 出典：[report/2026-06-17_edm_cifar10_s_churn_sensitivity_report.md](report/2026-06-17_edm_cifar10_s_churn_sensitivity_report.md)、`S_churn=6, 7`の追加は[report/2026-06-27_additional_report_s_churn_6_7.md](report/2026-06-27_additional_report_s_churn_6_7.md)。

この「学習不足のモデルほどサンプリング側の工夫が効く」という観察が、実験8（Restart）の動機になった。

## 構成

このグループは番号付きサブ実験ではなく、stage単位で構成されている。

| パス | 内容 |
|---|---|
| `stage0/` | 小規模な3×2の確認、4 checkpoint×S_churn、clip下での細かいS_churn |
| `stage1/` | FID-5kでのスクリーニング（48条件）、S_churn 6・7 |
| `stage2/` | FID-50kでの代表条件確認と追加2 seed範囲の評価 |
| [pndm/](pndm/README.md) | 同じS_churn感度をPNDM samplerで評価 |
| `docs/` | 診断指標メモ |
| `report/` | 報告書と図 |

## 未解決・次の候補

- PNDM版（[pndm/](pndm/README.md)）の結論は報告書にまとめられていない（要確認）。
