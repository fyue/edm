# 春学期に実施した内容（特別研究報告）

**EDMの再現実験**

春学期では、拡散モデルの理解を深めるために、教員の指導のもとで、EDM（Elucidating the Design Space of Diffusion-Based Generative Models）の実験に取り組みました。CIFAR-10のclass-conditional生成を対象として、学習、画像生成、FID評価までの流れを確認しました。

まず、公式のpretrained modelを使って画像生成とFID計算を確認しました。3つのseed範囲で評価したところ、FIDはおよそ1.84程度となり、論文の値である1.79に近い結果が得られました。

**自前学習と条件調整**

次に、CIFAR-10 class-conditional EDMを自分の環境で学習させました。GPUはQuadro RTX 8000を1枚使用し、途中のFIDを確認しました。

実験を進める中で、学習を長く続ければ必ずFIDが良くなるわけではなく、120k以降では悪化する場合があることが分かりました。そこで、160k checkpointを起点としてlearning rateやEMAの条件を変えました。その結果、learning rateを0.0002、EMAを0.5として170kまで継続した条件で、FID平均1.85124が得られました。これは今回の自前学習の中では最も良い結果でした。

**S_churnに関する実験**

また、EDM samplerのパラメータである`S_churn`が生成品質に与える影響についても調べました。各checkpointで`S_churn`を変えてFIDを比較しました。

結果として、学習初期のcheckpointほど`S_churn`による改善が大きく、学習が進むにつれて改善幅は小さくなりました。例えば、5.5kではFIDが約0.89改善しましたが、120kでは約0.026程度で、pretrained modelではほとんど改善が見られませんでした。このことから、まだ十分に学習されていないモデルではsamplingの設定が結果に影響する可能性があると考えました。

**今後の予定**

今後は、PNDM/S-PNDMなど別サンプラーとの比較などを、教員と相談しながら進める予定です。
