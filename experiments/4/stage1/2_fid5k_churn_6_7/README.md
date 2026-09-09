# Experiment 4 Stage 1 Second: S_churn=6,7 FID-5k Extension

## 目的

Stage 1 実験1の `S_churn=0-5` を変更せず、`S_churn=6,7` を追加してFID-5kがさらに下がるか確認する。

新規実行は次の16条件のみである。

| checkpoint | S_churn |
|---|---|
| `5.5k` | `6, 7` |
| `10k` | `6, 7` |
| `15.0k` | `6, 7` |
| `20k` | `6, 7` |
| `40k` | `6, 7` |
| `80k` | `6, 7` |
| `120k` | `6, 7` |
| `pretrained` | `6, 7` |

## 既存実験との分離

実験1の以下のファイルは読み取りにのみ使用し、変更しない。

```text
experiments/4/stage1/1_fid5k_selected_churn/
```

実験2の画像、ログ、生の結果は、すべてこのフォルダ内に保存する。統合結果も実験2の `results/` に新規生成する。

## 実行条件

- samples: `5000`
- seeds: `0-4999`
- steps: `18`
- sampler: EDM sampler（Euler予測 + Heun型2次補正）
- `S_min=0`
- `S_max=inf`
- `S_noise=1`
- FID reference: `fid-refs/cifar10-32x32.npz`
- 条件数: `16`
- 推定時間: 約3時間（既存Stage 1の約11分/条件から算出）

`steps=18` では `gamma=min(S_churn/18, sqrt(2)-1)` なので、`S_churn=6,7` はどちらもクリップ前であり、異なるsampling条件になる。

## 実行方法

予定だけ確認する。

```bash
PLAN_ONLY=1 experiments/4/stage1/2_fid5k_churn_6_7/scripts/run_fid5k_churn_6_7.sh
```

tmuxで本実行する。

```bash
tmux new -s edm-exp4-stage1-churn67
experiments/4/stage1/2_fid5k_churn_6_7/scripts/run_fid5k_churn_6_7.sh
```

進捗確認。

```bash
tail -f experiments/4/stage1/2_fid5k_churn_6_7/logs/fid5k-selected-churn.log
```

特定checkpointだけ実行する場合。

```bash
CHECKPOINT_FILTER="40k 80k 120k pretrained" \
  experiments/4/stage1/2_fid5k_churn_6_7/scripts/run_fid5k_churn_6_7.sh
```

実行後、実験1との統合表とグラフは自動生成される。手動で再生成する場合は次を実行する。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/4/stage1/2_fid5k_churn_6_7/scripts/merge_and_plot_fid5k_churn.py
```

## 出力

実験2の生データ:

```text
images/<checkpoint>/S_churn-{6,7}/n5000-seeds-0-4999/
logs/fid5k-*.log
results/fid5k_selected_churn_summary.tsv
```

実験1と実験2を統合した結果:

```text
results/fid5k_churn_0_7_combined.tsv
results/fid5k_churn_0_7_best.tsv
results/fid5k_churn_0_7_curve.png
```

低FID側の `40k, 80k, 120k, pretrained` だけを抽出した結果:

```text
results/fid5k_churn_0_7_low_fid.tsv
results/fid5k_churn_0_7_curve_low_fid.png
```

## 再開と停止

同じコマンドを再実行すると、実験2のsummaryに記録済みの条件はスキップする。

現在の条件終了後に停止する場合:

```bash
touch experiments/4/stage1/2_fid5k_churn_6_7/STOP_AFTER_CURRENT_CONDITION
```

## 解釈上の注意

- FID-5kはスクリーニング指標なので、小さい差は追加seedまたはFID-50kで確認する。
- `S_churn=6,7` で改善が続いても、`steps=18` では約 `7.46` でchurn量が飽和するため、それより上は同一条件になる。
- 実験1と同じ画像数、seed範囲、step数、`S_min`, `S_max`, `S_noise` を維持して比較する。
