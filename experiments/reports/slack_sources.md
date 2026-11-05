# Slack参照索引

研究に関する話題・日付・実験IDから、保存した原文と添付を探すための索引。参照元は2026-11-03取得の記録であり、その後のSlack更新は反映していない。

## 読み方と保存先

- 実験条件・数値は各実験READMEと`results/`で確認する。議論の経緯は、下表の原本でTSを検索し、前後のやり取りと添付も確認する。各表は日時順、日時はJST。
- 話題と実験IDは検索用に付けたラベルであり、発言本文の引用や結論・推奨方針ではない。索引の選択だけで分析を限定せず、必要に応じて原本全体を確認する。
- 完全な保存記録はローカル専用の[slack_origin/README.md](../slack_origin/README.md)、[DM原本](../slack_origin/dm.md)、[スレッド原本](../slack_origin/thread.md)。本文・raw JSON・添付を含めてGit対象外とし、原本の内容・配置は維持する。
- 原本がないPCでは、下表の「Slack」から直接参照する（閲覧権限が必要）。添付の相対リンクは原本のあるPCで利用できる。原本の共有・バックアップはこの索引では行わない。
- 保存範囲・取得上の制限は原本README、添付の出典・サイズ・SHA256は[attachment_manifest.json](../slack_origin/attachment_manifest.json)、取得件数などは[export_summary.json](../slack_origin/export_summary.json)を参照する。本文内の表は原本にあり、独立した添付ファイルがない場合もある。
- この索引には研究に関するDMと指定スレッドを掲載する。アカウント設定・認証情報・個人的な連絡は転記しない。
- 報告内容とローカル結果が一致しない場合は、両方の出典を保持し、未照合として扱う。Slackの報告値をローカルの評価データへ補完・上書きしない。

## 話題から探す

- [S1：研究テーマ・EDM再現](#s1)
- [S2：学習率・EMA・S_churn](#s2)
- [S3：サンプラー比較・Flow Matching・Restart導入](#s3)
- [S4：checkpoint系列・K sweep・データ数の相談](#s4)
- [S5：t_min比較・N_main固定・NFE固定](#s5)
- [S6：進捗スレッド・3 seed・200k以降・30k subset](#s6)
- [S7：Kの傾向図に関するやり取り](#s7)

## 実験IDから探す

| 実験ID | 話題表 |
|---|---|
| 1・2 | [S1：EDM再現](#s1) |
| 3・4 | [S2：学習率・EMA・S_churn](#s2) |
| 5・7 | [S3：サンプラー・FM](#s3) |
| 0-1・8-4 | [S4：checkpoint・K sweep](#s4) |
| 0-2・0-5・8-14 | [S4：データ数の相談](#s4)、[S6：30k subset報告](#s6)、[S7：返信](#s7)（0-5の独立した結果報告は本索引にない） |
| 8-8・8-10・8-11・8-12 | [S5：t_min・固定条件](#s5)、[S6：進捗](#s6)、[S7：図](#s7) |
| 0-3・0-4・8-13 | [S6：3 seed・200k以降](#s6) |

<a id="s1"></a>

## 研究テーマ・EDM再現

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-05-14 21:39:22 | 研究テーマの相談 | 全体 | [dm.md](../slack_origin/dm.md) · TS `1778762362.156099` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1778762362156099) | ファイル添付なし |
| 2026-05-15 21:47:31 | 研究テーマへの返信・EDM再現の相談 | 全体・1 | [dm.md](../slack_origin/dm.md) · TS `1778849251.068109` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1778849251068109) | ファイル添付なし |
| 2026-05-22 22:06:53 | 学習進捗の報告 | 1 | [dm.md](../slack_origin/dm.md) · TS `1779455213.801889` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1779455213801889) | ファイル添付なし |
| 2026-05-22 23:41:32 | 学習進捗への返信 | 1 | [dm.md](../slack_origin/dm.md) · TS `1779460892.312029` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1779460892312029) | ファイル添付なし |
| 2026-06-01 20:57:39 | EDM再現条件と学習経過の相談 | 1 | [dm.md](../slack_origin/dm.md) · TS `1780315059.014389` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1780315059014389) | ファイル添付なし |
| 2026-06-01 21:52:09 | kimg・再開方法の確認 | 1 | [dm.md](../slack_origin/dm.md) · TS `1780318329.830929` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1780318329830929) | ファイル添付なし |
| 2026-06-01 22:30:46 | kimg・再開方法の回答 | 1 | [dm.md](../slack_origin/dm.md) · TS `1780320646.894579` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1780320646894579) | ファイル添付なし |
| 2026-06-03 10:19:40 | EDM再現実験の報告 | 1 | [dm.md](../slack_origin/dm.md) · TS `1780449580.142149` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1780449580142149) | ファイル添付なし |
| 2026-06-03 14:12:35 | 条件付き生成の確認依頼 | 1・2 | [dm.md](../slack_origin/dm.md) · TS `1780463555.269409` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1780463555269409) | ファイル添付なし |
| 2026-06-04 15:20:13 | 条件付き生成の確認報告 | 2 | [dm.md](../slack_origin/dm.md) · TS `1780554013.812799` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1780554013812799) | ファイル添付なし |

<a id="s2"></a>

## 学習率・EMA・S_churn

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-06-11 17:08:23 | 学習率・EMA分岐の報告 | 3-1 | [dm.md](../slack_origin/dm.md) · TS `1781165303.478149` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1781165303478149) | ファイル添付なし |
| 2026-06-15 21:00:03 | 学習率探索への返信 | 3 | [dm.md](../slack_origin/dm.md) · TS `1781524803.838449` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1781524803838449) | ファイル添付なし |
| 2026-06-16 10:37:00 | EMA比較の報告 | 3-1 | [dm.md](../slack_origin/dm.md) · TS `1781573820.720519` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1781573820720519) | ファイル添付なし |
| 2026-06-18 09:51:23 | 継続学習・S_churn実験の報告 | 3・4 | [dm.md](../slack_origin/dm.md) · TS `1781743883.845349` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1781743883845349) | [1: S_churn 感度解析の実験結果について.pdf](../slack_origin/attachments/F0BATNQ5NET_S_churn%20%E6%84%9F%E5%BA%A6%E8%A7%A3%E6%9E%90%E3%81%AE%E5%AE%9F%E9%A8%93%E7%B5%90%E6%9E%9C%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6.pdf) |
| 2026-06-19 22:15:30 | S_churn調査への返信 | 3・4 | [dm.md](../slack_origin/dm.md) · TS `1781874930.498839` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1781874930498839) | ファイル添付なし |
| 2026-06-20 16:42:53 | 160k以降の学習率比較の報告 | 3 | [dm.md](../slack_origin/dm.md) · TS `1781941373.019999` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1781941373019999) | ファイル添付なし |
| 2026-06-21 22:07:25 | 継続学習への返信 | 3 | [dm.md](../slack_origin/dm.md) · TS `1782047245.888649` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1782047245888649) | ファイル添付なし |
| 2026-06-23 00:30:53 | 170k評価の報告 | 3 | [dm.md](../slack_origin/dm.md) · TS `1782142253.252749` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1782142253252749) | ファイル添付なし |
| 2026-06-24 21:51:03 | 170k評価への返信 | 3 | [dm.md](../slack_origin/dm.md) · TS `1782305463.148109` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1782305463148109) | ファイル添付なし |
| 2026-07-03 21:58:01 | 160k起点の学習率探索の報告 | 3-10 | [dm.md](../slack_origin/dm.md) · TS `1783083481.096039` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1783083481096039) | [1: image.png](../slack_origin/attachments/F0BEK6VMAPR_image.png) |
| 2026-07-05 23:12:18 | 学習率探索への返信 | 3 | [dm.md](../slack_origin/dm.md) · TS `1783260738.872859` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1783260738872859) | ファイル添付なし |

<a id="s3"></a>

## サンプラー比較・Flow Matching・Restart導入

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-07-15 09:00:04 | PNDM系サンプラーの報告 | 5 | [dm.md](../slack_origin/dm.md) · TS `1784073604.257429` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1784073604257429) | ファイル添付なし |
| 2026-07-18 19:11:42 | PNDM・RK4・線形多段法についての返信 | 5 | [dm.md](../slack_origin/dm.md) · TS `1784369502.153179` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1784369502153179) | ファイル添付なし |
| 2026-08-05 09:00:07 | PNDM・RK・AB比較の報告 | 5 | [dm.md](../slack_origin/dm.md) · TS `1785888007.796369` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1785888007796369) | [1: PNDM、RK、ABサンプラーの比較実験.pdf](../slack_origin/attachments/F0BMYNBNPCM_PNDM%E3%80%81RK%E3%80%81AB%E3%82%B5%E3%83%B3%E3%83%97%E3%83%A9%E3%83%BC%E3%81%AE%E6%AF%94%E8%BC%83%E5%AE%9F%E9%A8%93.pdf) |
| 2026-08-06 22:19:21 | サンプラー比較への返信・関連論文 | 5 | [dm.md](../slack_origin/dm.md) · TS `1786022361.695919` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1786022361695919) | ファイル添付なし |
| 2026-08-28 11:56:03 | EDM実験の考察資料 | 5 | [dm.md](../slack_origin/dm.md) · TS `1787885763.881219` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1787885763881219) | [1: 2026-08-28.pdf](../slack_origin/attachments/F0BTCATP7LH_2026-08-28.pdf) |
| 2026-09-02 00:01:39 | 考察資料への返信 | 5 | [dm.md](../slack_origin/dm.md) · TS `1788274899.787799` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1788274899787799) | ファイル添付なし |
| 2026-09-10 09:31:39 | Flow Matching実装・実験の進捗 | 7 | [dm.md](../slack_origin/dm.md) · TS `1789000299.151299` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789000299151299) | ファイル添付なし |
| 2026-09-11 23:00:31 | Restart論文と比較実験についての返信 | 7・8 | [dm.md](../slack_origin/dm.md) · TS `1789135231.491879` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789135231491879) | ファイル添付なし |
| 2026-09-14 09:00:00 | FM比較・Restart実験の進捗 | 7・8 | [dm.md](../slack_origin/dm.md) · TS `1789344000.803729` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789344000803729) | [1: image.png](../slack_origin/attachments/F0C1LTC5Q2Y_image.png)、[2: image.png](../slack_origin/attachments/F0C1CJVM8JZ_image.png) |
| 2026-09-17 14:04:57 | 追加実験の報告資料 | 5・7・8 | [dm.md](../slack_origin/dm.md) · TS `1789621497.066799` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789621497066799) | [1: 2026-09-19_実験報告_report.pdf](../slack_origin/attachments/F0C2L0T5XTN_2026-09-19_%E5%AE%9F%E9%A8%93%E5%A0%B1%E5%91%8A_report.pdf) |

<a id="s4"></a>

## checkpoint系列・K sweep・データ数の相談

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-09-18 00:03:20 | K sweep・均一checkpointについての返信 | 0-1・8-4 | [dm.md](../slack_origin/dm.md) · TS `1789657400.863009` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789657400863009) | ファイル添付なし |
| 2026-09-18 10:23:04 | Restart設定・checkpoint一覧の回答 | 0-1・8-4 | [dm.md](../slack_origin/dm.md) · TS `1789694584.977249` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789694584977249) | ファイル添付なし |
| 2026-09-22 10:22:36 | 再学習の開始報告 | 0-1 | [dm.md](../slack_origin/dm.md) · TS `1790040156.263249` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790040156263249) | ファイル添付なし |
| 2026-09-26 12:12:57 | 学習量と学習データ数についての相談 | 0・8 | [dm.md](../slack_origin/dm.md) · TS `1790392377.153739` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790392377153739) | ファイル添付なし |
| 2026-09-28 18:24:33 | K sweepの途中報告 | 8-4 | [dm.md](../slack_origin/dm.md) · TS `1790587473.085369` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790587473085369) | [1: image.png](../slack_origin/attachments/F0C5RUTPGDN_image.png) |
| 2026-09-28 20:19:08 | K sweepへの返信 | 8-4 | [dm.md](../slack_origin/dm.md) · TS `1790594348.030149` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790594348030149) | ファイル添付なし |
| 2026-10-01 22:12:24 | 学習期間・FID評価条件の質問 | 0・8 | [dm.md](../slack_origin/dm.md) · TS `1790860344.466059` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790860344466059) | ファイル添付なし |
| 2026-10-02 09:28:22 | 追加checkpointとFID評価の進捗 | 0-1・8-4 | [dm.md](../slack_origin/dm.md) · TS `1790900902.246339` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790900902246339) | [1: image.png](../slack_origin/attachments/F0C5YTT5AR1_image.png) |
| 2026-10-02 23:23:14 | subset学習期間・FID評価についての返信 | 0-2・8 | [dm.md](../slack_origin/dm.md) · TS `1790950994.360789` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790950994360789) | ファイル添付なし |

<a id="s5"></a>

## t_min比較・N_main固定・NFE固定

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-10-05 18:48:00 | t_min・時刻対応・比較条件の確認 | 8-8・8-10〜8-12 | [dm.md](../slack_origin/dm.md) · TS `1791193680.732089` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791193680732089) | ファイル添付なし |
| 2026-10-06 02:36:00 | 時刻対応・t_min比較の報告 | 8-8 | [dm.md](../slack_origin/dm.md) · TS `1791221760.398859` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791221760398859) | 本文内の表（原本参照） |
| 2026-10-06 10:22:14 | t_minを変更する実験についての質問 | 8-8・8-11 | [dm.md](../slack_origin/dm.md) · TS `1791249734.829559` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791249734829559) | [1: restart_sampling_progress_report.pptx](../slack_origin/attachments/F0C7T6Y014G_restart_sampling_progress_report.pptx) |
| 2026-10-06 21:47:04 | t_minのFID-50k比較報告 | 8-8 | [dm.md](../slack_origin/dm.md) · TS `1791290824.736699` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791290824736699) | [1: image.png](../slack_origin/attachments/F0C7XNK5QN4_image.png)、[2: image.png](../slack_origin/attachments/F0C7XNKT1R6_image.png)、[3: image.png](../slack_origin/attachments/F0C73760GSW_image.png) |
| 2026-10-06 22:37:21 | t_min比較への返信 | 8-8 | [dm.md](../slack_origin/dm.md) · TS `1791293841.705399` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791293841705399) | ファイル添付なし |
| 2026-10-07 15:16:48 | 170kのt_min比較報告 | 8-8 | [dm.md](../slack_origin/dm.md) · TS `1791353808.400729` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791353808400729) | [1: image.png](../slack_origin/attachments/F0C7F3S6GE8_image.png) |
| 2026-10-07 17:47:36 | 170k比較・実験構成への返信 | 8-8・8-11・8-12 | [dm.md](../slack_origin/dm.md) · TS `1791362856.017499` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791362856017499) | ファイル添付なし |
| 2026-10-07 21:34:40 | N_main固定実験の開始報告 | 8-11 | [dm.md](../slack_origin/dm.md) · TS `1791376480.690789` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791376480690789) | ファイル添付なし |
| 2026-10-12 22:06:06 | NFE固定時のactual_t_minの質問・本文内表 | 8-12 | [dm.md](../slack_origin/dm.md) · TS `1791810366.837159` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791810366837159) | 本文内の表（原本参照） |
| 2026-10-13 22:03:28 | NFE固定実験の位置づけへの返信 | 8-11・8-12 | [dm.md](../slack_origin/dm.md) · TS `1791896608.521359` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1791896608521359) | ファイル添付なし |
| 2026-10-15 11:39:46 | N_main固定実験の完了報告 | 8-11 | [dm.md](../slack_origin/dm.md) · TS `1792031986.938329` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1792031986938329) | [1: image.png](../slack_origin/attachments/F0C97PTM2A1_image.png)、[2: image.png](../slack_origin/attachments/F0C9CT0SS1Y_image.png)、[3: image.png](../slack_origin/attachments/F0C9F09K17T_image.png) |
| 2026-10-15 23:07:02 | 図とFID評価への返信 | 8-11 | [dm.md](../slack_origin/dm.md) · TS `1792073222.757459` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1792073222757459) | ファイル添付なし |
| 2026-10-20 18:34:11 | NFE固定実験の完了報告 | 8-12 | [dm.md](../slack_origin/dm.md) · TS `1792488851.061659` · [Slack](https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1792488851061659) | [1: image.png](../slack_origin/attachments/F0CASGK10KY_image.png) |

<a id="s6"></a>

## 進捗スレッド・3 seed・200k以降・30k subset

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-10-17 12:21:56 | 進捗報告・スレッド親メッセージ | 0・8 | [thread.md](../slack_origin/thread.md) · TS `1792207316.483529` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1792207316483529) | [1: 研究進捗1017.pdf](../slack_origin/attachments/F0C9JGZDT8X_%E7%A0%94%E7%A9%B6%E9%80%B2%E6%8D%971017.pdf) |
| 2026-10-19 23:15:49 | 散布図・3 seed・継続学習についての返信 | 0-3・8-11〜8-14 | [thread.md](../slack_origin/thread.md) · TS `1792419349.323939` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1792419349323939) | ファイル添付なし |
| 2026-10-21 10:47:08 | 散布図・NFE固定実験などの回答 | 0-3・8-11〜8-13 | [thread.md](../slack_origin/thread.md) · TS `1792547228.012519` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1792547228012519) | [1: image.png](../slack_origin/attachments/F0CASHFR2A2_image.png)、[2: image.png](../slack_origin/attachments/F0CBP52D5PS_image.png) |
| 2026-10-22 22:11:10 | 200k以降の評価報告・本文内表 | 0-3 | [thread.md](../slack_origin/thread.md) · TS `1792674670.089809` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1792674670089809) | 本文内の表（原本参照） |
| 2026-10-22 22:36:07 | 散布図・継続学習への返信 | 0-3・8-11・8-12 | [thread.md](../slack_origin/thread.md) · TS `1792676167.958939` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1792676167958939) | ファイル添付なし |
| 2026-10-27 00:17:27 | 6 checkpointの3 seed評価完了報告・本文内表 | 8-13 | [thread.md](../slack_origin/thread.md) · TS `1793027847.627969` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793027847627969) | 本文内の表（原本参照） |
| 2026-10-27 00:27:16 | 200k・210kの3生成seed範囲報告・本文内表 | 0-4 | [thread.md](../slack_origin/thread.md) · TS `1793028436.537769` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793028436537769) | 本文内の表（原本参照） |
| 2026-10-27 00:44:36 | 30k subsetのRestart実験開始報告・本文内表 | 0-2・8-14 | [thread.md](../slack_origin/thread.md) · TS `1793029476.722379` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793029476722379) | 本文内の表（原本参照） |
| 2026-11-01 11:41:55 | 30k subsetの学習・Restart評価の進捗報告 | 0-2・8-14 | [thread.md](../slack_origin/thread.md) · TS `1793500915.020579` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793500915020579) | [1: image.png](../slack_origin/attachments/F0CDV9K9TRQ_image.png)、[2: image.png](../slack_origin/attachments/F0CDVAM0FD0_image.png)、[3: image.png](../slack_origin/attachments/F0CDWQB606R_image.png) |

<a id="s7"></a>

## Kの傾向図に関するやり取り

| 日時（JST） | 話題 | 関連実験ID | 原本・TS・Slack | 添付 |
|---|---|---|---|---|
| 2026-11-02 17:41:27 | Best K近傍の平均を用いた図の提案 | 8-11・8-12 | [thread.md](../slack_origin/thread.md) · TS `1793608887.039959` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793608887039959) | [1: image.png](../slack_origin/attachments/F0CE24N6TFC_image.png)、[2: image.png](../slack_origin/attachments/F0CEWM3M0FJ_image.png) |
| 2026-11-02 17:52:59 | 重み付け平均を用いた図の提案 | 8-11・8-12 | [thread.md](../slack_origin/thread.md) · TS `1793609579.204429` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793609579204429) | [1: image.png](../slack_origin/attachments/F0CE0D7BXFY_image.png)、[2: image.png](../slack_origin/attachments/F0CE0D93N14_image.png) |
| 2026-11-02 23:13:42 | 30k subsetの解釈・図の提案への返信 | 8-11・8-12・8-14 | [thread.md](../slack_origin/thread.md) · TS `1793628822.297069` · [Slack](https://kato-ai-lab.slack.com/archives/C082JD35X5W/p1793628822297069) | ファイル添付なし |

添付リンクは実際の保存ファイル名に合わせている。PNDM・RK・AB比較PDFは、原本Markdownのリンク名と保存名で濁点のUnicode表現が異なるため、索引側で保存名へ合わせた（原本は未変更）。

## 索引の更新

原本を追加取得した場合は取得範囲を確認し、必要な研究メッセージについて話題・実験ID・日時・TS・Slack直接リンク・関連添付を追記する。原本の本文は転載せず、TSと添付パスを照合する。索引を更新しても原本の取得時点と現在の実験状態は分けて記録する。
