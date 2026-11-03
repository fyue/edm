# Slack取得テキスト（slack_dm_older）

取得日: 2026-11-03 JST。Slackコネクターが返した本文・添付メタデータ。添付バイナリの中身はこのファイルに含まない。

Channel: DM (D08LCS5CPM3)

=== Message from 加藤恒彦（教員） <tsunehiko.kato@rikkyo.ac.jp> (U06T5PPQBRP) at 2026-09-28 20:19:08 JST === 
Message TS: 1790594348.030149
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790594348030149
大変良い結果だと思います。K依存性のばらつきが大きいですが、大雑把にいうと、学習が進む → 最適なK は小さくなる傾向（80kはイレギュラー）のように見えますね。
全体の走査が終わったら、先日の training checkpoint と FID の関係のプロットも、Restart については minimum の値を使ったものにアップデートしましょう。
Reactions: 承知しました (1), saluting_face (1)

=== Message from キョウキンキン(M2) <25vr041p@rikkyo.ac.jp> (U08LCS51HPF) at 2026-09-28 18:24:33 JST === 
Message TS: 1790587473.085369
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790587473085369
お世話になっております。

最適Kの走査実験について、再学習した10k～80kの各チェックポイントで、K=0～8をFID-5kにより実験していました。現在の状況はご報告します。

すべてのチェックポイントでK=0より低いFIDが得られましたが、最適Kは順に7, 6, 3, 4, 5, 5, 1, 7となり、30k以降学習のほど単調には減少しませんでした。

よろしくお願いいたします。
Reactions: +1 (1)
Files: image.png (ID: F0C5RUTPGDN, image/png, 350.1 KB)

=== Message from 加藤恒彦（教員） <tsunehiko.kato@rikkyo.ac.jp> (U06T5PPQBRP) at 2026-09-26 12:12:57 JST === 
Message TS: 1790392377.153739
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790392377153739
この「学習不足とSDE/ODEの関係」の調査が終わったら、次は「サンプル数が少ない場合のSDE/ODEの関係」を調べてみると良いと思います。（「学習不足」と「サンプル数が少ない」は同じではないので）

現在は CIFAR-10 の 50k 枚すべてのデータを使っていると思いますが、使える学習用データ数が少ない場合はどうなるか、ということです。
使うデータを 10k、20k、30k、40k とした場合をそれぞれ考え、それぞれで同様に K の依存性を見るとよいと思います（学習はそれぞれでフルに行う）。これも学習にはある程度時間がかかると思うので、順番は 30k -> 10k -> 20k -> 40k のような感じで、余裕が無いときはすべてを実行できなくてもいいと思います。
Reactions: 承知しました (1), saluting_face (1), thanks (1)

=== Message from キョウキンキン(M2) <25vr041p@rikkyo.ac.jp> (U08LCS51HPF) at 2026-09-22 10:22:36 JST === 
Message TS: 1790040156.263249
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1790040156263249
おはようございます。
厳密に再学習実験を改めて設定して実行し始めたことをご報告いたします。順調であれば、来週週末に終わる見込みです。
よろしくお願いいたします。
Reactions: +1 (1)

=== Message from キョウキンキン(M2) <25vr041p@rikkyo.ac.jp> (U08LCS51HPF) at 2026-09-18 10:23:04 JST === 
Message TS: 1789694584.977249
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789694584977249
お忙しいところご確認いただきありがとうございます。

各チェックポイントでKを変えて走査して最適なKを求め、学習ステップ数との関係を確認することについて、承知しました。Restartの実験条件は一点伝え漏れてすみませんでした。t_maxからt_minへ戻るODEを、何個の時刻点で離散化するかを表す「N_restart」というパラメーターがあります。3に固定して実験していましたけど、Kと学習ステップ数の関連性の確認ですので、次の実験でも3に固定して対応させていただきます。

チェックポイントについて、厳密な10k倍数として不足しているのは13点です。ただし、以下のようにこれら13点にも非常に近い既存チェックポイントがあります。最大のズレでも0.106kでございますが、厳密値が必須であれば再学習を行います。

| 目標 | 既存snapshot |
|---:|---:|
| 30k | 30.035k |
| 50k | 50.035k |
| 60k | 60.070k |
| 70k | 70.106k |
| 90k | 90.035k |
| 100k | 100.070k |
| 110k | 110.106k |
| 130k | 130.035k |
| 140k | 140.070k |
| 150k | 150.106k |
| 170k | 170.035k |
| 180k | 180.070k |
| 190k | 190.106k |

また、論文のご紹介をいただきありがとうございます。
確認させていただきます。

引き続きどうぞよろしくお願い致します。

=== Message from 加藤恒彦（教員） <tsunehiko.kato@rikkyo.ac.jp> (U06T5PPQBRP) at 2026-09-18 00:03:20 JST === 
Message TS: 1789657400.863009
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789657400863009
大変良い結果だと思います。特にRestart の結果は、当初の「学習不足またはサンプル数が少ない場合に改善策はあるか」という問いに対する1つの答えになりうるので、重要だと思います。この結果の考察もしておくと良いです。

今回の結果は Restart で K=5 固定だと思いますが、さらに各チェックポイントで K=0, 1, 2, ... 5, .. 8 ぐらいまで走査して最適な K を求め、その最適な K と 学習ステップ数との関係を図にしてみると良いと思います。（おそらく、学習ステップが進むにつれて K が減少していき、あるところで K=0 つまり純粋な ODE に至る、という進化が見られるのではないかと思います）

また、現在のテスト用チェックポイントが 5.5k, 20k, 120k, 200k とかなり不均一なので、修論本番用に 10k か 20k ごとにチェックポイントを残すように再学習ができればさらに良くなると思います。

以下の論文も参考になるかもしれないので、読んでおくとよいでしょう。
<https://arxiv.org/abs/2306.02063|arxiv.org/abs/2306.02063>

=== Message from キョウキンキン(M2) <25vr041p@rikkyo.ac.jp> (U08LCS51HPF) at 2026-09-17 14:04:57 JST === 
Message TS: 1789621497.066799
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789621497066799
この２週間でやっていた実験状況をご報告します。
よろしくお願いいたします。
Reactions: +1 (1)
Files: 2026-09-19_実験報告_report.pdf (ID: F0C2L0T5XTN, application/pdf, 647.2 KB)

=== Message from キョウキンキン(M2) <25vr041p@rikkyo.ac.jp> (U08LCS51HPF) at 2026-09-14 09:00:00 JST === 
Message TS: 1789344000.803729
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789344000803729
ありがとうございます。
FMは20kまで学習させ、5k, 20kの比較実験をしてみました。結果は添付画像でご確認いただけます。
また、Restartは、論文を読み終え、サンプリング実験を設定しております。結果が出次第、ご報告いたします。
よろしくお願いいたします。
Reactions: +1 (1)
Files: image.png (ID: F0C1LTC5Q2Y, image/png, 171.9 KB), image.png (ID: F0C1CJVM8JZ, image/png, 147.3 KB)

=== Message from 加藤恒彦（教員） <tsunehiko.kato@rikkyo.ac.jp> (U06T5PPQBRP) at 2026-09-11 23:00:31 JST === 
Message TS: 1789135231.491879
Source: https://kato-ai-lab.slack.com/archives/D08LCS5CPM3/p1789135231491879
FMを実行している間に、以下の論文を読んでみると良いと思います。
「学習不足のモデルからのサンプリングで品質を改善できるか」という観点では、参考になる論文の1つだと思います。
（FMの実行が終わったら）EDMのいくつかのチェックポイントで、Restartと通常のODE(Heun)でのサンプリング品質を比較してみても良いと思います。
<https://arxiv.org/abs/2306.14878v2|arxiv.org/abs/…>


取得範囲: Restart関連の2026年9月11日以降のみ抜粋（それ以前の別テーマは保存対象外）。

