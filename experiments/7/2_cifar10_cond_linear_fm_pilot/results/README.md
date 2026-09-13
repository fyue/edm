# Runtime results

`run_fm_linear_pilot.sh`が次のTSVを生成する。

- `fm_linear_pilot_health.tsv`: stageごとのloss、sec/kimg、GPU memory、checkpoint。
- `fm_linear_pilot_fid_summary.tsv`: FID、sampler、steps、NFE、seed範囲、画像path。

数値を修論で使う前に、対応するcheckpoint・seed範囲・画像枚数と併せて確認する。
