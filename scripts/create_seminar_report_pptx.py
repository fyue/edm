#!/usr/bin/env python3
"""Create the integrated seminar report deck for 2026-07-25."""

from __future__ import annotations

from create_pndm_analysis_pptx import (
    SlideBuilder,
    build_pptx_from_slides,
    slide1 as pndm_slide1,
    slide2 as pndm_slide2,
    slide3 as pndm_slide3,
)


OUTPUT = "experiments/reports/seminar_report_2026-07-25_edm_pndm.pptx"
TRAINING_IMAGE = "experiments/3/report/image/2026-07-25/1784936878727.png"


def training_slide() -> SlideBuilder:
    slide = SlideBuilder("160k以降の追加学習実験")
    slide.textbox(
        0.65,
        1.05,
        7.45,
        0.45,
        ["160k checkpointから学習率を変えて継続学習した結果"],
        16,
        "333333",
        fill="EEF5F5",
        line_color="D9E2E2",
        radius=True,
    )
    slide.picture(0.65, 1.65, 7.45, 4.25, TRAINING_IMAGE)
    slide.textbox(
        8.45,
        1.25,
        4.1,
        0.5,
        ["見るポイント"],
        18,
        "0F3D46",
        True,
        align="ctr",
    )
    slide.textbox(
        8.35,
        1.95,
        4.35,
        3.3,
        [
            "FID約2.05の160k checkpointから小さい学習率で継続",
            "160kから170k付近まではFIDが下がる傾向",
            "170k以降まで続けるとFIDが再び高くなる傾向",
            "lr=1e-4は上がり幅が比較的小さく、追加確認中",
        ],
        16,
        "222222",
        bullet=True,
    )
    return slide


def main() -> None:
    slides = [training_slide(), pndm_slide1(), pndm_slide2(), pndm_slide3()]
    build_pptx_from_slides(OUTPUT, slides, "2026-07-25 ゼミ報告")


if __name__ == "__main__":
    main()
