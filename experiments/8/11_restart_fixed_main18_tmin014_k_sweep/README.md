# 実験8-11：N_main=18・t_min=0.14固定のK sweep

主実験。0-1の20k〜200kでN_main=18・t_min=0.14に固定してK=0〜8を走査する。FID-50kの最適Kは20k:7、40k〜120k:3、140k以降は主に1で、学習が進むほど小さい。

実験0-1のCIFAR-10 class-conditional EDM（DDPM++、学習seed=0）について、主経路を18 stepsに固定し、Restart回数Kと学習量の関係を比較する。計算量はKに応じて変化する。実験8-10のNFE固定比較とは異なる条件である。

## 再現条件

- checkpoint：実験0-1の20k〜200k、20k刻みの10点。kは学習量kimg（1kimg=1,000枚）。config TSVにモデルパスを記載。
- K=0〜8、各checkpointで9条件。全体90条件、FID-5k/25k/50kの270評価。
- 全条件N_main=18、EDM Heun。K≥1ではN_restart=3、t_min=0.14、t_max=0.30、restart_noise=1。K=0はRestartなし（config上のRestart steps、t_min/maxは0、実開始sigma/indexはnull）。
- rho=7、sigma_min=0.002、sigma_max=80、S_churn=0。モデルの対応範囲でsigmaをクリップする。
- t_minはEDM主スケジュールの最寄りのsigmaへ対応する。恒等round_sigmaのEDMPrecondではactual_t_min≈0.139516468731、restart_index=13（0始まり）。計画・protocol・結果に記録する。
- seed：5kは0〜4999、25kは0〜24999、50kは0〜49999。全条件で同じseed集合を使用し、ラベルは生成CLIのseedに対応するランダムクラス。
- 各PCで1 GPU、生成batch=64、FID batch=64。参照統計は`fid-refs/cifar10-32x32.npz`。
- 固定seed集合1組の比較であり、統計的有意差は主張しない。異なる画像数のFIDは同じステージ内で比較する。

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NFE = 35 + 4K | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 | 67 |

## 実行とcheckpoint指定

リポジトリ直下で実行する。edm環境をactivateして`python`・`torchrun`を使うか、`PYTHON=/path/to/edm/bin/python`を指定する。torchrunはそのPythonと同じbinディレクトリから選択する（`TORCHRUN`で変更可能）。

```bash
conda activate edm
RUN=experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh

# GPU不要、書き込みなし：90条件・270評価・最大450万枚を確認
"$RUN" --plan-only --fid-stage all

# checkpoint指定：単一、列挙、範囲、間隔付き範囲
"$RUN" --checkpoint-filter '40k' --plan-only
"$RUN" --checkpoint-filter '20k,60k 100k' --plan-only
"$RUN" --checkpoint-filter '20k-140k' --plan-only
"$RUN" --checkpoint-filter '20k-140k:40k' --plan-only

# 固定seedの小規模生成（GPU待機あり、FID評価なし）
"$RUN" --samples-only --sample-seeds 0-1 --checkpoint-filter 20k

# 指定ステージまでの未完了評価を順に実行
"$RUN" --fid-stage 5000
"$RUN" --fid-stage 25000
"$RUN" --fid-stage 50000
"$RUN" --fid-stage all

# 条件の絞り込み、検証、集計
"$RUN" --checkpoint-filter 20k --condition-filter '20k-k0,20k-k8' --plan-only
"$RUN" --verify-only --fid-stage all
"$RUN" --aggregate-only
```

`--checkpoint-filter`の未指定または`all`は全10 checkpoint。範囲は両端を含み、間隔の既定は20k。間隔は正の20kの倍数で、終点に一致する必要がある。重複は除去し、checkpointは学習量順に実行する。対象外・逆順・不正な間隔・空の条件選択はエラー。モデルの存在確認・SHA256計算は選択したcheckpointだけを対象とするため、別PC担当のモデルは不要。

CLI指定は対応環境変数より優先する。`FID_STAGE`（既定5000）、`CHECKPOINT_FILTER`、`CONDITION_FILTER`、`EXP_ROOT`、`CONFIG`、`FID_REF`、`GEN_BATCH`、`FID_BATCH`、`PLAN_ONLY`、`VERIFY_ONLY`、`AGGREGATE_ONLY`、`SAMPLES_ONLY`、`SAMPLE_SEEDS`、`RESUME`に対応する。`all`は全選択条件の5k→全選択条件の25k→全選択条件の50kの順で実行する。

GPU処理は`run.sh`が`scripts/load_gpu_config.sh`をsourceし、共通`scripts/wait_for_gpu_and_run.sh`を経由する。GPU番号・GPU名は各PCの同期対象外`~/.config/edm/gpu.env`に記載し、共有コードへ固定しない。`GPU_INDEX`・`EXPECTED_GPU_NAME`が必須。別設定は`EDM_GPU_CONFIG`で指定する。ローカル設定が環境変数より優先され、別GPUに切り替えない。確認間隔は`CHECK_INTERVAL_SEC`。help・plan・verify・aggregateはGPU設定不要。

## 2台での分担と統合

両PCで同じ設定・モデル・コード・FID参照統計を使用し、出力先を分ける。GPUが異なる場合も同じbatch設定を維持する。実行中の共通集計ファイルやログを2台から更新しない。

```bash
# こちら：7 checkpoint、63条件、189評価
EXP_ROOT=experiments/8/11_restart_fixed_main18_tmin014_k_sweep/runs/20k-140k \
experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh \
  --checkpoint-filter '20k-140k' --fid-stage all

# もう一台：3 checkpoint、27条件、81評価
EXP_ROOT=experiments/8/11_restart_fixed_main18_tmin014_k_sweep/runs/160k-200k \
experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh \
  --checkpoint-filter '160k-200k' --fid-stage all

# 両出力先の条件別完了JSONがこちらに揃った後に統合集計
experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh \
  --aggregate-only --result-roots \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/runs/20k-140k \
  experiments/8/11_restart_fixed_main18_tmin014_k_sweep/runs/160k-200k
```

転送が必要なのは各出力先の`results/<checkpoint>-k<K>-fid<count>.json`だけ。画像・モデル・ログ・参照統計は集計に不要。転送先も`<出力先>/results/`構造を保つ。統合結果は`EXP_ROOT`または`--exp-root`の`results/`へ保存され、未指定なら実験ルートの`results/`へ保存される。統合した一覧JSONは完了マーカーとして再取り込みしない。

統合では実験条件、seed、batch、同一checkpointのモデルSHA256、参照統計とgenerate.py/fid.py/ランナーのSHA256を照合する。各JSONにprotocolを含めるため、画像の転送は不要。PC固有のパス・GPU名・ファイル更新日時の差は許容する。同一条件・ステージの結果が複数ある場合は、値が一致していてもエラー。担当範囲が重複しないようにする。

## 画像・再開・結果

条件ごとの`images/<condition>/pool/`に最大50k枚を保持する。25kではseed 5000〜24999、50kでは25000〜49999を追加し、中断後は不足seedだけ生成する。ステージ別のFID用ビューはハードリンク、別ファイルシステムではコピーする。他実験の画像・FIDは自動取り込みしない。スモーク画像はpoolと別管理する。

resumeは既定で有効。モデル・参照統計・コードのSHA256と生成条件が変わった場合は停止する。完了評価もprotocolと画像集合を検証してスキップする。破損・余分なseedはエラー。`--no-resume`は既存画像・結果があれば停止する。各出力先の`logs/.run.lock`で同一PC上の同一出力先への多重実行を防ぐ。停止するには出力先に`STOP_AFTER_CURRENT_CONDITION`を置く。現在の条件・ステージの保存を終えて停止し、再開前にファイルを取り除く。

- `results/<condition>-fid<count>.json`：完了マーカー。FID、seed、生成条件、protocolとID、実際の開始sigma/index、GPU名、生成枚数、生成/FID/画像処理を含む評価経過時間を保存。
- `restart_fid_summary.tsv/.json`：完了結果一覧（TSVでは詳細protocolを省略、JSONに保持）。
- `fid_by_checkpoint_and_k.tsv`：ステージ別FIDとFID(K)−FID(0)。負なら改善。
- `optimal_k.tsv`：9候補が揃ったcheckpoint・ステージの最適K。同値なら小さいK。
- `fixed_main18_tmin014_fid{5000,25000,50000}.{png,svg}`：FID–K曲線、差分ヒートマップ、学習量と最適K。
- `time_estimate.json`：各PCの担当範囲の残り時間見積もり。

生成時間は完了評価を保存した実行での追加生成時間。中断した過去の実行時間は含まない。生成画像・ログ・キャッシュ・checkpointはコミットしない。

## 所要時間の見込み

Quadro RTX 8000相当を各PCで1枚使い、全画像を新規生成する前提。実験8-8のt_min=0.14・K=5で、5k生成約11〜20分、追加20k約44〜78分、追加25k約55〜97分という実測から、生成を画像数×NFEに比例すると仮定して換算した概算。

| 対象 | GPU待機を除く目安 |
|---|---:|
| 1台で全90条件の5k | 約22〜30時間 |
| 全90条件を25kへ拡張 | 追加約70〜115時間 |
| 全90条件を50kへ拡張 | 追加約85〜145時間 |
| 1台で全条件、画像検証等を含む | 約8〜14日 |
| こちら：20〜140k | 約6〜10日 |
| もう一台：160〜200k | 約3〜5日 |
| 同程度の2台で並行した全体 | 約6〜10日 |

各PCで最初に全KのFID-5kが完了したcheckpointを基準とし、生成時間/画像数/NFE、FID時間/画像数、画像処理時間/画像数の中央値から未完了評価の残り時間を推定する。各評価後に`ESTIMATE`として表示し、`results/time_estimate.json`へ保存する。生成済みの途中画像、中断・GPU待機、結果転送、plot生成時間は見積もりに含めない。GPUやI/O速度、FIDの固定起動コストで変動するため目安として扱う。

## CPU検証

```bash
python -m compileall -q experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts
bash -n experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh
bash -n experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/schedule_after_gpu_free.sh
python experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/test_experiment.py
```

テストの生成・FIDは模擬処理であり、一時ディレクトリのみへ書き込む。小さなCPU mock networkでK=0〜8の実NFEも確認する。本番GPUジョブは別途実行する。
