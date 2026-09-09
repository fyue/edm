# PNDM Stage 2

Stage 2はStage 1で選択した条件をFID-50kで確認し、代表的な
`S_churn=0` 対 selected churnのペアを追加seedで検証する。

初期configにはEDM版と直接対応する候補を置いている。PNDM Stage 1の結果を確認せずに
本実行しないこと。

`config/reused_experiment5_pretrained.tsv` はExperiment 5で計算済みの
pretrained、PNDM、`S_churn=0`、18 stepsの3 seed結果であり、再生成対象ではない。
