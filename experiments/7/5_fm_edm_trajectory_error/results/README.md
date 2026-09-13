# Results

`pilot/`と`final/`は実行時に作成される。各stageにはseedごとの相対L2と経路形状、
集計TSV、SVG図、実行条件、完了markerを保存する。

判定では、まずRK4 coarse/fine間の`reference_relative_l2`が各solverの誤差より
十分小さいことを確認する。その後、同一model内でNFEによる誤差減少を読み、最後に
FM/EDM間で無次元の誤差・直線性指標を比較する。
