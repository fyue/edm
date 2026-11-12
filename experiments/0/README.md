# 実験0：修論用の学習系列（均一checkpoint・データ量・200k以降）

## 問い

修論の解析で使う学習済みモデル系列を用意する。50,000枚の全データで10k刻みのcheckpointを持つ系列を作り、30,000枚・10,000枚のsubsetでも学習して、データ量がFIDの推移に与える影響を調べる。200k以降も継続学習して改善するかを調べる。

## 主要な結論

- **0-1**：全データ、seed 0、10k〜200kの10k刻みcheckpoint（[checkpoint_manifest.tsv](1_cifar10_cond_edm_10k_retraining/results/checkpoint_manifest.tsv)）。実験8-4・8-6・8-10〜8-13の評価対象モデル。
- **0-2（30k subset）**：FID-50k（seed 0-49999）は 10k 5.42342、20k 2.81792、30k 2.52083、40k 3.09847、**50k 2.34864（最良）**、60k 2.75990、その後100k 4.44395まで悪化。約50k以降の悪化は過学習的な挙動と解釈している。出典：[fid_summary.tsv](2_cifar10_cond_edm_30k_subset/results/fid_summary.tsv)。
- **0-3（200k以降）**：K=0のFIDは 200k 1.80678、210k 1.83710、220k 2.00929。220kではK=1が2.05315で悪化（seed 0-49999の1範囲。出典：[comparison.tsv](3_cifar10_cond_edm_post200k_extension/results/comparison.tsv)）。
- **0-4**：200k・210kでのK=1の3 seed評価。Slackの[2026-10-27報告](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793028436537769)では小さな改善が報告されている（[原本・TSの索引](../reports/slack_sources.md#s6)）。ただしこのPCには初回seed範囲の4評価しかなく（n=1、K=1の改善は200k +0.01343、210k +0.01809。出典：[improvement_stats.tsv](4_post200k_restart_k1_fid50k_3seeds/results/improvement_stats.tsv)）、最終値として使わない（元PCの結果と要照合）。
- **0-5（10k subset）**：30,000 kimgまで完了。FID-50kは10k 4.63680、20k 2.82997（最良）、30k 3.77162（[fid_summary.tsv](5_cifar10_cond_edm_10k_subset/results/fid_summary.tsv)、[run_status.tsv](5_cifar10_cond_edm_10k_subset/results/run_status.tsv)）。
- **0-6（30k subset、連続学習）**：10kのsnapshot/stateが保存済み、100k完了は未確認（2026-11-05）。0-2は再開のたびに`--seed=0`で同じノイズ・augmentation・画像順を再生しており（0-1との差に再生の有無が混ざる）、これを切り分ける。FIDへの影響は未確定。出典：[0-6 README](6_cifar10_cond_edm_30k_subset_continuous/README.md)。他の実験を含む一覧は[乱数再生の監査](../reports/rng_replay_audit.md)を参照。

- **0-7（200k→300k）**：0-1の200kから未使用seedで継続し、210k〜300kの10点を作成する。起動後はGPU待機・学習中として扱い、300k完了は検証結果で確認する。出典：[0-7 README](7_cifar10_cond_edm_post200k_no_replay/README.md)。FID・Restart走査は後続実験とする。

## 実験一覧と役割

| ID | 役割 |
|---|---|
| [0-1](1_cifar10_cond_edm_10k_retraining/README.md) | 全データ・10k刻みの再学習。Restart実験の基準モデル系列 |
| [0-2](2_cifar10_cond_edm_30k_subset/README.md) | 30,000枚subsetの学習系列。8-14の対象 |
| [0-5](5_cifar10_cond_edm_10k_subset/README.md) | 10,000枚subsetの学習系列。30kまで完了、20kがFID最良（[結果](5_cifar10_cond_edm_10k_subset/results/fid_summary.tsv)） |
| [0-6](6_cifar10_cond_edm_30k_subset_continuous/README.md) | 0-2と同じ3万枚を0-1と同じく連続学習する系列。再生の影響の切り分け |
| [0-3](3_cifar10_cond_edm_post200k_extension/README.md) | 0-1の200kから210k・220kへの継続学習とK=0/K=1の比較 |
| [0-4](4_post200k_restart_k1_fid50k_3seeds/README.md) | 0-3の200k・210kでのK=1の3 seed評価 |
| [0-7](7_cifar10_cond_edm_post200k_no_replay/README.md) | 0-1の200kから300kへ未使用seedで追加学習し、後続評価用checkpointを作成・検証 |

## 未解決・次の候補

- 0-4の3 seed集計を元PCの結果と照合する。
- 0-6の結果が出たら、0-2の悪化と8-14の結論を見直す。
- 完了した0-5について、8-14と同様のRestart比較を行うかを決める。
