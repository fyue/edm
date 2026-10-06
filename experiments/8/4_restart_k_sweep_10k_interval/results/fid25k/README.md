# Experiment 8-4 FID-25k Results

実行時に選択した各checkpointで`K=0,...,8`をFID-25kにより評価する。
未指定時はFID-5kの9条件が完了したcheckpointを自動選択する。
seed `0-4999`は既存FID-5k画像を再利用し、seed `5000-24999`を追加生成する。

実行後に次を生成する。

- `reuse_manifest.tsv`: 既存画像の再利用元・出力先・方式・枚数
- `restart_fid_summary.tsv`: 条件ごとのFID-25kと再現条件
- `restart_k_sweep.tsv`: checkpoint × KのFIDとK=0との差
- `optimal_k.tsv`: checkpointごとのFID最小K
- `fid_by_checkpoint_and_k.{tsv,md}`: K=0～8の横持ち表
- `restart_k_sweep_overview.{png,svg}`: 全体概要
- `fid_vs_k_by_checkpoint.{png,svg}`: checkpoint別FID–K曲線
- `fid_delta_vs_k0_heatmap.{png,svg}`: K=0との差のヒートマップ
