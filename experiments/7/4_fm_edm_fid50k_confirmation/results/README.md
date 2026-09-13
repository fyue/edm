# Results

`fid50k_confirmation.tsv`は実行時に作成され、各条件について以下を記録する。

- model、sampler、solver、steps、NFE
- FID-5k（Experiment 7-3）とFID-50k
- 生成・FID計算時間
- checkpoint、画像directory、ログ、FID参照統計
- 新規生成、既存生成物、再利用画像の区別

画像とログは大容量のためGit管理対象外とする。
