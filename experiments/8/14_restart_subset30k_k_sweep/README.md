# 実験8-14：30,000枚学習の20k〜100kにおけるRestart K比較

0-2（30,000枚学習）の20k〜100kで8-11型・8-12型のK sweepを行い、全データ学習の結果と比べる。実行中（2026-11-03時点、100kのFID-50kは未完了）。

実験0-2（CIFAR-10 class-conditional、DDPM++/EDM、学習seed=0、subset seed=0、30,000枚）の保存済みcheckpointを使う。追加学習は行わない。ベースラインFIDが悪いほど同じKでRestart改善幅が大きいかを調べ、50,000枚学習の実験8-11・8-12と比較する。

## 再現条件

| 項目 | main18（8-11準拠） | nfe55（8-12準拠） |
|---|---|---|
| checkpoint | 20k・40k・60k・80k・100k（学習量kimg） | 同左 |
| K | 0〜8 | 0〜8 |
| N_main | 18 | 28−2K |
| NFE | 35＋4K | 55 |

K=0はEDM Heun。K≥1はRestart Heun、N_restart=3、t_min=0.14、t_max=0.30、restart_noise=1。rho=7、sigma_min=0.002、sigma_max=80、S_churn=0。実開始sigma/indexは主スケジュールへの最近傍対応で記録する。nfe55ではN_mainに伴って実開始sigmaも変化する。

各PCで1 GPU、生成/FID batch=64。全CIFAR-10の`fid-refs/cifar10-32x32.npz`を使用する。FID-5k/25k/50kはそれぞれseed 0–4999/24999/49999で、同じseedに対応するランダムクラスラベルを使う。poolに不足seedのみ追加する。90条件・270評価・最大450万枚。両系列で同一条件となるK=5も別評価とし、他実験の画像・FIDを自動流用しない。

## 実行

リポジトリ直下、edm環境で実行する。別環境は`PYTHON=/path/to/edm/bin/python`を指定する。torchrunは同じbinディレクトリから選択する。

```bash
EXP=experiments/8/14_restart_subset30k_k_sweep
# GPU不要・無書き込み。両系列それぞれ45条件・135評価・225万枚。
"$EXP/scripts/run_all.sh" --plan-only
"$EXP/scripts/run.sh" --mode main18 --help
# 全条件の小規模生成→両系列の5k→25k→50k→比較集計
"$EXP/scripts/schedule_after_gpu_free.sh"
# 個別選択・再開
"$EXP/scripts/run.sh" --mode nfe55 --checkpoint-filter '20k,40k,60k' --fid-stage all
"$EXP/scripts/run.sh" --mode main18 --samples-only --sample-seeds 0-1
"$EXP/scripts/run.sh" --mode main18 --verify-only --fid-stage all
"$EXP/scripts/run.sh" --mode main18 --aggregate-only
python "$EXP/scripts/compare_results.py" --require-complete
```

`--mode main18|nfe55`はrun.shの先頭引数とする（既定main18）。Python直接実行では`FIXED_MODE`環境変数を使用する。出力先は`runs/main18`と`runs/nfe55`。`--exp-root`で個別に変更できるが、比較スクリプトは標準出力先を読む。全体実行では固定の2出力先を使用する。計画表示・検証・集計はGPU設定不要。全体シーケンスと各出力先はflockで重複実行を拒否する。

GPU処理は`scripts/load_gpu_config.sh`と共通`scripts/wait_for_gpu_and_run.sh`を経由する。`~/.config/edm/gpu.env`のGPU_INDEX・EXPECTED_GPU_NAMEが必須。別設定はEDM_GPU_CONFIGで指定する。ローカル設定を優先し、別GPUへ切り替えない。待機間隔はCHECK_INTERVAL_SEC（既定300秒）。

80k・100kも `--checkpoint-filter '80k,100k'` で選択できる。選択したモデルが未配置の場合は、GPU処理を始める前にエラーになる。片方しか同期されていない場合は、配置済みcheckpointのみを指定する。未指定・all・run_all.shは追加した80k・100kも含む。

既存の互換登録がある出力先にこの対象追加を適用する場合は、次のCPU専用操作を一度実行する。

```bash
PYTHON=/path/to/edm/bin/python
"$PYTHON" experiments/8/14_restart_subset30k_k_sweep/scripts/extend_checkpoint_compatibility.py \
  --exp-root experiments/8/14_restart_subset30k_k_sweep/runs/main18
```

旧登録のコード写しと新版全体をASTで比較し、checkpoint定数と対応するエラーメッセージ以外が変わっていれば拒否する。実行ロック下で既存画像・結果を検証し、旧登録とコードを`runner_compatibility_history/`に保存して更新する。既存protocol ID・完了結果・画像は変更しない。その他のランナー変更を自動承認する操作ではない。nfe55に適用する場合は`FIXED_MODE=nfe55`と対応する出力先を指定する。

## 旧ランナーとの互換登録

即時中断方式の旧ランナーから条件境界待機方式へ更新した場合、ランナーSHA256の違いにより既存poolの検証が停止する。今回確認済みの旧版 `a5272ce13e72593fe6f3fda2485e1ab043e228606a6b91d59043b201be5bf443` に限り、出力先ごとに明示登録できる。

```bash
PYTHON=/path/to/edm/bin/python \
experiments/8/14_restart_subset30k_k_sweep/scripts/run.sh --mode main18 \
  --exp-root experiments/8/14_restart_subset30k_k_sweep/runs/main18 \
  --register-runner-compatibility
```

この操作はCPU専用で、実行ロックを取得し、全pool・staging・完了結果・進捗・FID画像ビューを検証する。Git indexの旧版SHA256とサンプリング・検証関数も照合する。モデル・生成/FIDコード・参照統計・batchなどが異なる場合は拒否する。旧版のGit indexが必要なのは初回登録時だけ。登録済みの同じコードで再登録しても既存ファイルは変更しない。

`<EXP_ROOT>/runner_compatibility/registration.json` と同ディレクトリの `*.source` に旧版・新版・互換処理・共通停止制御・GPU設定読込処理のコードを保存する。既存pool・結果・protocol IDは書き換えず、登録済み出力先の新規評価も旧版を互換基準として集計の整合性を維持する。実際の実行コードSHA256はGPU実行時の `<EXP_ROOT>/logs/runner-executions/*.json` に別記する。完了結果内の `protocol.runner` はこの出力先の互換基準であり、実際のランナーの識別は実行記録を参照する。

登録後にランナー、互換処理、共通停止・GPU制御、保存したコード写しが変われば再登録を含め拒否する。将来の変更を自動承認しない。登録は出力先の絶対パスにも結び付けるため、別PC・別出力先に移した登録はそのまま使用できない。`--plan-only`・`--verify-only`・`--aggregate-only` は登録を作成しない。通常の未登録実験に対する厳密なprotocol比較は維持する。

```bash
python experiments/8/14_restart_subset30k_k_sweep/scripts/test_runner_compatibility.py
```

## 停止・自動再開

既定の絶対パスは`/data/kyou/STOP_RUNNING_PASCAL`。各PCでそのPCのファイルを監視する。

```bash
touch /data/kyou/STOP_RUNNING_PASCAL  # 現在の条件の生成・FID完了後に待機
rm /data/kyou/STOP_RUNNING_PASCAL     # GPUの空きを再確認して自動再開
```

親ランナー/tmuxは起動したままにする。8-13と同じ共通`scripts/condition_pause.py`の`ConditionPause`を使い、次の未完了条件の開始前にSTOP待機とGPU確認を行う。実行中は子プロセスを中断せず、**現在のcheckpoint・K・FIDステージの生成、画像検証、FID、結果保存・集計まで完了する**。例えば5k中ならその条件の5k評価まで完了し、次の条件へ進む前に待機する。小規模生成は現在のcheckpoint・Kの指定seed集合の生成・検証を完了する。

完了済み評価はprotocolと画像を検証してスキップし、全対象完了済み・最終条件完了後はSTOPがあっても待機せず正常終了する。STOPが途中で削除され、条件境界で存在しなければ待機しない。削除後は共通GPU待機スクリプトで設定GPUの名前・空きを再確認して次の未完了条件へ進む。GPU待機中にSTOPが再作成された場合も開始直前に再確認する。待機中も出力先のロックを保持する。

- `PAUSE_ENABLED=0`：STOP待機とETA監視を無効化。GPU設定・空き確認は維持する。
- `PAUSE_FILE=/absolute/path`：監視先。既定`/data/kyou/STOP_RUNNING_PASCAL`。
- `PAUSE_POLL_SEC`：STOP検知間隔、既定1秒。
- `PAUSE_ETA_UPDATE_SEC`：ETA追記間隔、既定30秒。段階変更時も追記する。

既存CLIの`--pause-file`・`--pause-poll-sec`も維持し、環境変数より優先する。全体シーケンスには環境変数を指定する。help・plan・verify・aggregateは監視・STOP追記・GPU確認を行わない。

8-13と同じ`scripts/pause_eta.py`が、既存STOPファイルに実験名（main18/nfe55）、ホスト・PID・起動ID、checkpoint、K、seed範囲、評価枚数、現在の段階、条件完了までの見積もりを追記する。`tail -n 12 /data/kyou/STOP_RUNNING_PASCAL`で確認可能。履歴不足は「推定中」、超過は「見積もり超過・処理継続中」。排他制御し、削除したSTOPは再作成せず、symlinkや通常ファイル以外へは追記しない。追記失敗は警告として待機を維持する。

各系列の`logs/master.log`と`logs/pause.log`に待機・再開、`logs/pause-runtime.<PID>.json`にETA状態、`logs/pause-timings.jsonl`に画像検証・保存時間を記録する。

親runnerを終了させるには、該当系列の`runs/main18/STOP_AFTER_CURRENT_CONDITION`または`runs/nfe55/STOP_AFTER_CURRENT_CONDITION`を置く。終了用STOPを優先し、現在の条件は完了する。待機中やGPU確認中も終了用STOPに対応する。削除後に同じ実行コマンドで再開する。子プロセス失敗はエラー終了する。

## 結果と解釈

各系列の`results/`に条件別完了JSON、restart_fid_summary.tsv/.json、fid_by_checkpoint_and_k.tsv、optimal_k.tsv、FID曲線・差分ヒートマップ・最適KのPNG/SVG、残り時間見積もりを出力する。モデル・コード・FID参照のSHA256とprotocolを保存し、再開時の混在を拒否する。

実験ルートの`results/`には系列別の比較TSVと3パネルPNG/SVGを出力する。50,000枚学習の同じ20k・40k・60kとのFID–K比較、subset改善幅、既存全checkpointのベースラインFID対改善幅への重ね合わせを掲載する。生成/FIDコードと参照統計のSHA256、seed・生成条件・batchの整合性を検証する。改善幅はFID(K=0)−FID(K)（正が改善）。各FID枚数・各固定条件を分けて比較する。

既存通常ODEのFID-50kは20k=2.81792、40k=3.09847、60k=2.75990。学習量順には単調でないため、同じKでベースラインFIDと改善幅の関係を見る。3 checkpoint、単一学習seed、固定生成seed集合の探索結果であり、傾向が揃わない場合も報告する。最適Kは探索内最良値で、統計的有意差やデータ不足の因果効果を主張しない。

## main18・FID図の選択表示

次のCPU専用スクリプトは、保存済み完了JSONを読み、20k・40k・60kだけを3パネルすべてに表示する。ヒートマップは対象データの最大絶対値による対称スケールで、負値は青、0は純白、正値は赤とする。

```bash
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/14_restart_subset30k_k_sweep/scripts/plot_main18_fid5k.py

# 25k・50kも同じ表示範囲・配色で再描画
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/14_restart_subset30k_k_sweep/scripts/plot_main18_fid5k.py --num-images 25000
/home/kyoukinkin/data/conda-envs/edm/bin/python \
  experiments/8/14_restart_subset30k_k_sweep/scripts/plot_main18_fid5k.py --num-images 50000
```

更新対象は`runs/main18/results/fixed_main18_tmin014_fid<枚数>.png`と同名SVGだけ。`--num-images`は5000（既定）・25000・50000から選び、指定しなかった枚数の図は変更しない。`--results-dir`で別の結果ディレクトリを指定できる。対象27件が不足・重複・不正な場合は描画前に拒否する。生結果・集計表・pool・protocol・互換登録は変更せず、GPU設定や生成・FID処理も使用しない。通常のランナー集計は全checkpointの図を再生成するため、その後も選択表示が必要ならこのスクリプトを再実行する。

```bash
python experiments/8/14_restart_subset30k_k_sweep/scripts/test_plot_main18_fid5k.py
```

## CPU検証

```bash
python -m compileall -q experiments/8/14_restart_subset30k_k_sweep/scripts
FIXED_MODE=main18 python experiments/8/14_restart_subset30k_k_sweep/scripts/test_experiment.py
FIXED_MODE=nfe55 python experiments/8/14_restart_subset30k_k_sweep/scripts/test_experiment.py
python experiments/8/14_restart_subset30k_k_sweep/scripts/test_comparison.py
python experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/test_experiment.py PauseTests
```

模擬生成/FIDと一時ディレクトリによる段階評価・再開、破損画像、protocol不一致、条件完了後の待機/削除後のGPU再確認、停止要求で子プロセスを終了しないこと、生成後の停止要求でもFID・結果保存が完了すること、完了済みスキップと最終条件で待機しないこと、原子的保存、全条件計画を検証する。CPU mock networkで両系列の実NFEを確認する。本番では全54条件のseed 0〜1を先に生成・検証する。生成画像・ログ・キャッシュ・checkpointはコミットしない。
