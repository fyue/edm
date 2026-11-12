# experiments/ の地図

CIFAR-10 class-conditional EDM（DDPM++）を中心とした実験の置き場。修論の主題は、**学習不足のモデルをサンプリング側の工夫（S_churn、Restart）で改善できるか**である。

## 読む順序

1. このファイル：グループの役割と系譜を把握する。
2. [INDEX.md](INDEX.md)：全実験の一覧（自動生成）。実験IDから目的とパスを引く。
3. `<グループ>/README.md`：グループの問い、主要な結論（数値と出典）、重要な実験。
4. `<グループ>/<実験>/README.md`：条件、実行方法、再現条件。数値は`results/*.tsv|json`で確認する。

議論の経緯は[Slack参照索引](reports/slack_sources.md)から原文と関連する添付を確認する。話題・日付・実験IDで探せる索引であり、特定の要約や解釈を分析より優先するものではない。原本の`slack_origin/`はローカルにのみ保持し、Gitには含めない。原本がないPCでは索引のSlack直接リンクを利用する。報告内容とローカル結果の不一致は、両方の出典を付けて未照合として扱う。

`training-runs/`、`fid-evals/`、`generated/`、`images/`、`logs/`は大容量の成果物なので読まない（Git対象外）。

## グループ

| グループ | テーマ | 状態 |
|---|---|---|
| [1-cifar10-...](1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/README.md) | 最初の本番学習（1 GPU、200k）。5.5k/20k/120k/200kのcheckpointは実験3・4・8-1〜8-3で使用 | 完了 |
| [2](2/README.md) | 120k/200kでの`--class`指定の動作確認 | 完了 |
| [3](3/README.md) | 160k以降のlr・EMA探索。最良は3-10の170k（FID 1.85124、3 seed平均） | 完了 |
| [4](4/README.md) | S_churn感度。学習初期ほど改善が大きい | 完了 |
| [5](5/README.md) | pretrainedでのsampler比較（PNDM・RK4・AB） | 完了 |
| [6](6/README.md) | 前向きノイズ付加後のノルム分布 | 完了 |
| [7](7/README.md) | Linear Flow MatchingとEDMの比較 | 完了 |
| [0](0/README.md) | 修論用の学習系列：全データの10k刻み（0-1）、30k/10k subset（0-2/0-5）、30k subsetの連続学習（0-6）、200k以降（0-3/0-4）、未使用seedによる300k延長（0-7） | 0-5は完了 |
| [8](8/README.md) | **Restart Sampling（修論の主実験）** | 8-14は実行中、8-15はcheckpoint監視 |

番号は作成順ではない。グループ0はグループ8の基準モデルを作るため、後から作成した。

## 系譜（どの成果物が何に使われるか）

```text
1-cifar10-... (本番学習 5.5k/20k/120k/200k)
 ├─ 120k/160k → 3-1 → 3-10 170k【実験3のbest】 → 8-7, 8-8
 │                   └─ 3-18 175k → 3-19, 3-20
 ├─ 各checkpoint → 4 (S_churn), 8-1〜8-3 (固定K=5)
 公式pretrained → 5 (sampler比較), 4, 8-7
0-1 (全データ 10k〜200k, 10k刻み)
 ├─ → 8-4, 8-6, 8-10, 8-11【主実験】, 8-12 → 8-13 (3 seed)
 ├─ 200k → 0-3 (210k/220k) → 0-4 (3 seed)
 └─ 200k → 0-7（未使用seedで300kまで追加学習、GPU待機・進捗は実験README参照）
0-2 (30k subset) → 8-14
0-6 (同じ30k subset、連続学習) → 8-15（checkpoint監視・直接FID-50k）→ 0-2の再開影響の切り分け
7-1 → 7-2 (FM 20k) → 7-3 → 7-4, 7-5
5-4, 5-11 → 5-13 (軌道距離)
```

## 報告・議論の記録

- [乱数再生の監査](reports/rng_replay_audit.md)：`--seed`固定の再開で乱数列を再生した学習実験と、そのcheckpointを評価した実験の一覧。
- [Slack参照索引](reports/slack_sources.md)：研究に関する原文・添付への入口（ローカル原本とSlack直接リンク）。
- [reports/](reports/)：横断的な報告（例：[2026-09-17 AB軌道・FM・Restart](reports/2026-09-17_additional_experiments_report.md)、[2026-10-02の計画](reports/2026-10-02.plan.md)）。
- `<グループ>/report/`：グループ別の報告。8は`report/<MMDD>/`の日付別。
- [spring_semester_report.md](spring_semester_report.md)：春学期のまとめ。

## 更新ルール

新しい実験を作ったり結果が確定したりしたら、次を行う。

1. 実験READMEのH1直後の段落に、目的（と確定後は結論）を1〜2文で書く。
2. グループREADMEの「主要な結論」「実験一覧と役割」を更新する。
3. `python scripts/build_experiment_index.py`でINDEX.mdを再生成する。
4. 系譜や進行中の状態が変わったら、このファイルを更新する。
