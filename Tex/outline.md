# 修士論文の題名と構成案

2026-11-05時点。Restartを中心に、加藤先生とのSlackでの議論と `experiments/` の実測値から作成した案。

議論の原文・添付は[Slack参照索引](../experiments/reports/slack_sources.md)、複数rootを統合した評価結果と出典は[11月7日報告の再現手順](../experiments/8/report/1107/README.md)を参照する。この構成案は執筆用の提案であり、原文や結果データより優先するものではない。

## 題名

**推奨：学習不足の拡散モデルに対するRestartサンプリングの効果 ― 学習量・学習データ量と最適Restart回数の関係 ―**

英題：Effects of Restart Sampling on Undertrained Diffusion Models: Dependence of the Optimal Restart Count on Training Amount and Dataset Size

代替案：

- 問いを前に出す：Restartサンプリングは学習不足とデータ不足を補えるか ― CIFAR-10 EDMにおける検証 ―
- 当初のテーマに寄せる：有限計算資源下の拡散モデルにおけるRestartサンプリングの学習段階依存性

「救えない」は題名に入れない。先生の解釈は30,000枚の1ケースに基づくため、結論の章で条件付きで述べる。

## 論文の位置づけ

中心の問いは、先生の発言（9/18、9/26）にある「学習不足またはサンプル数が少ない場合に改善策はあるか」である。学習不足とデータ不足は別の問題として扱う。

当初のテーマ（適応的sampler）は提案手法に至っていないので、新手法の提案ではなく、**Restartの効果がどんな条件で現れるかを調べた実証研究**として書く。

- RQ1：学習不足のモデルの生成品質を、Restartで改善できるか
- RQ2：最適なRestart回数Kは学習量にどう依存するか
- RQ3：学習データが少ない場合も同じことが成り立つか

## データで確認した主張

数値はFID-50k。改善幅は FID(K=0) − FID(K) で、正なら改善。

| # | 主張 | 根拠 | 強さ |
|---|---|---|---|
| 1 | baseline FIDが悪いほど改善が大きい | 8-11の10点で、K固定でも相関が高い（K=3でr=0.98、K=1で0.93）。最良Kではr=0.997だが、選択の効果を含む | 強い（学習seedは1つ） |
| 2 | 改善は生成seedを変えても再現する | 8-13：20k +0.381±0.013、60k +0.105±0.007。K選択に使ったseed範囲を除いても+0.374、+0.102 | 強い |
| 3 | 十分学習すると改善は誤差と区別しにくい | 180k +0.019±0.017、200k +0.013±0.012。3範囲とも正だが、対応ありt≈2（df=2）で有意ではない | 「有意な改善」と書かない |
| 4 | 最適Kは学習が進むと小さくなる | 8-11の近傍平均Kは7.5→1.5。最良K自体はガタつく（160kで3に戻るなど） | 中。K=0への到達は未確認（180k・200kはK=0も近傍に入る） |
| 5 | 改善はNFE増加だけでは説明できない | ODEをN=18→28に増やしてもFIDは改善しない（20k −0.018、200k +0.029） | 強い。NFE交絡への反論になる |
| 6 | NFE固定でも改善の傾向は同じ | 8-12：N=28のODEに対する最大の改善は20k +0.421、200k +0.025 | 中。Kの依存性は8-11より乱れる（近傍平均Kは7→1だが途中で上下）。Kでactual_t_minも変わる |
| 7 | **baseline FIDだけでは効果を予測できない** | 全データ20k（K=0で3.02）は最大+0.39だが、30k学習の80k（3.69）は全K≥1で悪化。40k（3.10）、60k（2.76）も同様。イレギュラーな40kを除いても成り立つ | 最も新しい主張。subset・学習seedが各1つ |
| 8 | 30k学習では効果が学習初期に限られる | 20kは全Kで改善（最大 main18 +0.165、nfe55 +0.200）、40k以降は全Kで悪化 | 中。FID最良の50kが未走査 |

主張7が論文の独自性の中心になる。「FIDが悪い＝Restartが効く」は学習不足の範囲でしか成り立たない。

### 主実験（8-11）の要約

N_main=18、t_min=0.14、seed 0-49999。近傍平均Kは、最良FIDとの差が0.02以下のKの単純平均（統計的な同等性の判定ではない）。

| checkpoint | K=0 | 最良K | 最良FID | 近傍平均K |
|---|---:|---:|---:|---:|
| 20k | 3.023 | 7 | 2.629 | 7.5 |
| 40k | 2.312 | 3 | 2.152 | 5.75 |
| 60k | 2.121 | 3 | 2.011 | 4 |
| 80k | 1.957 | 3 | 1.877 | 3 |
| 100k | 1.881 | 3 | 1.836 | 2.5 |
| 120k | 1.859 | 3 | 1.827 | 2 |
| 140k | 1.865 | 1 | 1.821 | 2.5 |
| 160k | 1.882 | 3 | 1.850 | 2 |
| 180k | 1.834 | 1 | 1.822 | 2 |
| 200k | 1.807 | 1 | 1.793 | 1.5 |

## 章構成

図のパスは `experiments/` からの相対パス。TeXに入れるときはPDFかPNGを `Tex/figures/` に置く。

### 第1章 序論

- 1.1 背景：有限計算資源と学習不足のモデル
- 1.2 問い：RQ1〜RQ3
- 1.3 貢献：学習量とRestart効果の関係の定量化、最適Kの学習量依存性、学習不足とデータ不足の違い
- 1.4 本論文の構成

### 第2章 背景と関連研究

- 2.1 拡散モデルとSDE/ODEによる定式化（Song et al., 2021）
- 2.2 EDM（Karras et al., 2022）
- 2.3 Restart Sampling（Xu et al., 2023）：ODEは離散化誤差が小さく、SDEの確率性は蓄積誤差を収縮させるという説明
- 2.4 ODEとSDEの優劣条件（Cao et al., 2023）：スコアの誤差が生成過程のどの時期に入るかで優劣が変わる
- 2.5 高速solver（PNDM、DPM-Solver、DEIS、DPM-Solver++）

### 第3章 実験設定

- 3.1 データセットとモデル：CIFAR-10 class-conditional、DDPM++、EDM
- 3.2 学習系列：全データ（0-1）、30,000枚（0-2）、200k以降（0-3）
- 3.3 EDMの再現：0-1の200kはFID 1.802（3範囲平均）。同じ手順の公式モデルは1.839、論文値は1.79
- 3.4 Restartの実装と設定：N_restart=3、(t_min, t_max)=(0.14, 0.30)、NFE=2N_main−1+4K
- 3.5 評価方法：FID-50k、3 seedは生成seed範囲で再学習ではない
- 3.6 指標：改善幅、最良K、近傍平均K

図：`0/2_cifar10_cond_edm_30k_subset/results/training_fid50k.png` は第6章で使う。

### 第4章 学習不足に対するRestartの効果（主実験）

RQ1・RQ2に答える章。先生の位置づけ（10/13）どおり、Restartの効果を見る実験とする。

- 4.1 KとFIDの関係、ヒートマップ：`8/11_restart_fixed_main18_tmin014_k_sweep/results/fixed_main18_tmin014_fid50000.png`
- 4.2 baseline FIDと改善幅の関係：`8/11_restart_fixed_main18_tmin014_k_sweep/results/fid_baseline_vs_improvement_fid50000.png`
- 4.3 最適Kの学習量依存性（単純平均の図）：`8/11_restart_fixed_main18_tmin014_k_sweep/results/fixed_main18_tmin014_fid50000_mean_k.png`
- 4.4 3 seedでの確認：`8/13_restart_optimal_k_fid50k_3seeds/results/fid50k_3seeds.png`。K選択に使った範囲を除いた値も示す
- 4.5 学習量とFIDの推移（Restartは最小値）：図は未作成
- 4.6 NFE増加の影響：ODEのstep増加では改善しないことを第5章の結果とあわせて示す

### 第5章 計算量固定での比較

「計算量固定＋Restart実装での挙動」を見る実験で、Kだけの効果ではないと明記する（10/13）。

- 5.1 NFE=55の設定とactual_t_minの表（`8/12_restart_fixed_nfe55_tmin014_k_sweep/README.md`）
- 5.2 結果：`8/12_restart_fixed_nfe55_tmin014_k_sweep/results/fixed_nfe55_tmin014_fid50000.png`
- 5.3 最適Kの学習量依存性：`8/12_restart_fixed_nfe55_tmin014_k_sweep/results/fixed_nfe55_tmin014_fid50000_mean_k.png`
- 5.4 N_main固定との比較：20kのような学習初期ではactual_t_minが小さいKが有利に見える点を、付録Aの結果と関連づける

### 第6章 学習データ不足の場合

RQ3に答える章。

- 6.1 30,000枚学習の学習曲線：50kで最良（2.349）、100kで4.444
- 6.2 Restartの効果：`8/14_restart_subset30k_k_sweep/runs/main18/results/fixed_main18_tmin014_fid50000.png`、`8/14_restart_subset30k_k_sweep/runs/nfe55/results/fixed_nfe55_tmin014_fid50000.png`
- 6.3 全データとの比較：baselineが同程度でも効果が逆になる点（主張7）
- 6.4 200k以降の全データ学習との比較（0-3、0-4）：210kはFIDが悪化しても改善、220kはK=1で悪化（1範囲のみ）

### 第7章 考察

- 7.1 Restartの効果を決める要因：FIDの悪さではなく、その原因が学習不足か過学習か
- 7.2 誤差の観点からの解釈：蓄積誤差の収縮（Xu et al.）と誤差の入る時期（Cao et al.）。**実験で確かめていない仮説として書く**
- 7.3 Kの実用的な選び方
- 7.4 限界：学習seedが1つ、CIFAR-10のみ、指標はFIDのみ、subsetは1ケース

### 第8章 結論

RQ1〜RQ3への答え。データ不足の結論は「今回の条件では」と限定する。

### 付録

- A. パラメータ選定：t_minの比較（`8/8_restart_tmin_comparison/results/restart_tmin_fid50000_lines.png`）。20k・40kでは0.06、60k・170kでは0.14が有利（先生の指示：10/6、10/7）
- B. 予備実験：固定K=5（`8/3_restart_k5_fid50k_3seeds/results/restart_k5_fid50k_3seed_overview.png`、8-1、8-2、8-6、8-7）、動機になったS_churn感度（`4/stage2/2_fid50k_extra_seeds/results/fid50k_3seed_improvement.png`）
- C. 周辺実験：学習率探索（実験3）、他のsampler（実験5）、Flow Matching（実験7）
- D. 再現条件

**系列の注意**：8-1〜8-3と実験4は実験1の系列（200kでFID 2.18）で、第4章以降の0-1系列（1.80）とは別のモデル。図や数値を混ぜない。両系列の差の原因は切り分けていないので断定しない。

## 先生の意見の反映先

| 日付 | 先生の発言 | 反映先 |
|---|---|---|
| 9/18 | Restartは問いへの答えになりうる | 第1章の動機 |
| 9/18 | 学習が進むとKが減りK=0に至るのでは | 第4章で検証。K=0到達は未確認と書く |
| 9/26 | 学習不足とサンプル数不足は別 | RQの分離、第6章を独立 |
| 9/28 | checkpointとFIDの図はRestartの最小値で更新 | 4.5（未作成） |
| 10/6 | パラメータ選定の節をAppendixに作る | 付録A |
| 10/7 | EDMの標準設定は(0.14, 0.30)。t_min依存性は発展的実験かAppendixに | 3.4、付録A |
| 10/13 | N_main固定がRestartの効果を見る実験、NFE固定は実装の挙動。実際のt_minの表を載せる | 第4章・第5章の位置づけ、5.1 |
| 10/15 | ヒートマップがわかりやすい。論文の図はFID-50k | 全図の方針 |
| 10/19 | 180k・200kは統計誤差の可能性 | 主張3、4.4 |
| 10/22 | FIDが悪いほど改善が大きい。210kは過学習でも改善する点が興味深い | 4.2、6.4 |
| 11/2 | データ不足はRestartでは救えない。40kはイレギュラー | 6.2、第8章で限定して記述 |
| 11/2 | Kの図は計算根拠が素直な平均値を使う | 4.3。指数重み付き平均は採用しない |

## 執筆前に埋める穴

### GPU不要

1. **3 seedの集計入口の統一**：実験直下の暫定集計では8-13の100k・140kと0-4がn=1のままだが、11/5に報告用スクリプトで複数rootから8-13の36/36評価・0-4の12/12評価を再集計できた。報告用スナップショットは全条件n=3である。本文で使う値は[集計元と再現手順](../experiments/8/report/1107/README.md)を明記し、Slack報告値との区別を保つ
2. **checkpointとFIDの図**：Restart側を最小値にして8-11から作る（4.5）
3. **8-14の比較集計**：`compare_results.py` の出力がまだない
4. **再開による乱数再生の影響**：[実験0-6](../experiments/0/6_cifar10_cond_edm_30k_subset_continuous/README.md)を0-2との比較に使う前に、単一プロセスで学習したかを起動履歴で確認する。0-2のFID悪化をデータ不足や過学習だけで説明できるかは未確定。再生の対象となる実験は[乱数再生の監査](../experiments/reports/rng_replay_audit.md)を参照

### GPU必要（推奨順）

時間はRTX 8000の実測（NFE 39で50,000枚が約1.4時間）からの概算。

4. **30k学習の50kと30kのK走査**：FID最良の50kが走査に入っていない。主張8の最大の穴で、40kのイレギュラーの判断にも効く。snapshotは保存済み。K={0,1,3,5,8}で約18時間
5. **8-14の残り**：main18の100k、nfe55の80k・100kのFID-50k。このPCでは実行中のプロセスがないので、別PCで動いているか確認する
6. **過学習の直接の確認**：抽出されなかった20,000枚と学習用30,000枚でdenoising lossを比べる。学習lossは下がり続けるだけで（100kで0.1626、全データは0.1634）、差が小さく根拠にならない。短時間で済む見込み
7. **220kの3 seed**：K=1の悪化（−0.044）は1範囲のみ。4評価、約5.5時間
8. **主要点の3 seed（任意）**：30k学習の20kと40k。主張7の土台を固める

### 不要

- 8-10：8-12が置き換えた
- 10k・20k・40kのデータ量実験：先生が1ケースでよいと明言（10/2）。0-5（10,000枚）は完了済みなので、余裕があれば付録の補足に使う

## 参考文献（`references.bib` に追加が必要）

現在は karras2022edm のみ。

- Song et al. (2021), Score-based generative modeling through stochastic differential equations
- Xu et al. (2023), Restart Sampling for Improving Generative Processes（arXiv:2306.14878）
- Cao et al. (2023), Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations（arXiv:2306.02063）
- PNDM（arXiv:2202.09778）、DPM-Solver（arXiv:2206.00927）、DEIS（arXiv:2204.13902）、DPM-Solver++（arXiv:2211.01095）
