#!/usr/bin/env python3
"""Create a small editable PPTX report for the EDM/PNDM seminar slides."""

from __future__ import annotations

import html
import os
import zipfile
from pathlib import Path


EMU = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000


def emu(inches: float) -> int:
    return int(inches * EMU)


def xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def paragraphs(lines, size=22, color="222222", bold=False, bullet=False, align=None):
    out = []
    for line in lines:
        attrs = []
        if align:
            attrs.append(f'algn="{align}"')
        if bullet:
            attrs.extend(['marL="342900"', 'indent="-171450"'])
        ppr_attr = (" " + " ".join(attrs)) if attrs else ""
        bullet_xml = '<a:buChar char="•"/>' if bullet else ""
        ppr_xml = f"<a:pPr{ppr_attr}>{bullet_xml}</a:pPr>" if attrs or bullet else ""
        bold_attr = 'b="1" ' if bold else ""
        out.append(
            f"<a:p>"
            f"{ppr_xml}"
            f'<a:r><a:rPr lang="ja-JP" sz="{size * 100}" '
            f'{bold_attr}>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="Yu Gothic"/><a:ea typeface="Yu Gothic"/>'
            f"</a:rPr><a:t>{xml_escape(line)}</a:t></a:r>"
            f"</a:p>"
        )
    return "".join(out)


class SlideBuilder:
    def __init__(self, title: str):
        self.shape_id = 2
        self.items = []
        self.media = []
        self.background = "FFFFFF"
        self.textbox(0.45, 0.25, 12.45, 0.55, [title], 24, "0F3D46", True)
        self.line(0.45, 0.88, 12.45, "0F3D46", 2)

    def next_id(self):
        self.shape_id += 1
        return self.shape_id

    def textbox(
        self,
        x,
        y,
        w,
        h,
        lines,
        size=20,
        color="222222",
        bold=False,
        fill=None,
        line_color=None,
        bullet=False,
        align=None,
        radius=False,
    ):
        sid = self.next_id()
        fill_xml = (
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            if fill
            else "<a:noFill/>"
        )
        line_xml = (
            f'<a:ln><a:solidFill><a:srgbClr val="{line_color}"/></a:solidFill></a:ln>'
            if line_color
            else "<a:ln><a:noFill/></a:ln>"
        )
        geom = "roundRect" if radius else "rect"
        self.items.append(
            f"<p:sp>"
            f"<p:nvSpPr><p:cNvPr id=\"{sid}\" name=\"Text {sid}\"/>"
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f"<p:spPr><a:xfrm><a:off x=\"{emu(x)}\" y=\"{emu(y)}\"/>"
            f"<a:ext cx=\"{emu(w)}\" cy=\"{emu(h)}\"/></a:xfrm>"
            f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
            f"{fill_xml}{line_xml}</p:spPr>"
            f'<p:txBody><a:bodyPr wrap="square" anchor="ctr" lIns="91440" '
            f'rIns="91440" tIns="45720" bIns="45720"/><a:lstStyle/>'
            f"{paragraphs(lines, size, color, bold, bullet, align)}</p:txBody>"
            f"</p:sp>"
        )

    def line(self, x, y, w, color="222222", width=1):
        sid = self.next_id()
        self.items.append(
            f"<p:cxnSp><p:nvCxnSpPr><p:cNvPr id=\"{sid}\" name=\"Line {sid}\"/>"
            f"<p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>"
            f"<p:spPr><a:xfrm><a:off x=\"{emu(x)}\" y=\"{emu(y)}\"/>"
            f"<a:ext cx=\"{emu(w)}\" cy=\"0\"/></a:xfrm>"
            f'<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:ln w="{width * 12700}"><a:solidFill><a:srgbClr val="{color}"/>'
            f"</a:solidFill></a:ln></p:spPr></p:cxnSp>"
        )

    def picture(self, x, y, w, h, path: str):
        sid = self.next_id()
        rel_id = f"rId{len(self.media) + 2}"
        self.media.append((rel_id, path))
        self.items.append(
            f"<p:pic>"
            f"<p:nvPicPr><p:cNvPr id=\"{sid}\" name=\"Picture {sid}\"/>"
            f"<p:cNvPicPr><a:picLocks noChangeAspect=\"1\"/></p:cNvPicPr>"
            f"<p:nvPr/></p:nvPicPr>"
            f"<p:blipFill><a:blip r:embed=\"{rel_id}\"/>"
            f"<a:stretch><a:fillRect/></a:stretch></p:blipFill>"
            f"<p:spPr><a:xfrm><a:off x=\"{emu(x)}\" y=\"{emu(y)}\"/>"
            f"<a:ext cx=\"{emu(w)}\" cy=\"{emu(h)}\"/></a:xfrm>"
            f"<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom>"
            f"<a:ln><a:solidFill><a:srgbClr val=\"D9E2E2\"/>"
            f"</a:solidFill></a:ln></p:spPr>"
            f"</p:pic>"
        )

    def table(self, x, y, col_widths, row_h, rows, header_fill="0F3D46"):
        fills = [header_fill] + ["F4F7F7", "FFFFFF"] * 8
        for r, row in enumerate(rows):
            cur_x = x
            for c, text in enumerate(row):
                is_header = r == 0
                fill = fills[0] if is_header else fills[1 + (r % 2)]
                color = "FFFFFF" if is_header else "222222"
                size = 15 if is_header else 16
                self.textbox(
                    cur_x,
                    y + row_h * r,
                    col_widths[c],
                    row_h,
                    [text],
                    size=size,
                    color=color,
                    bold=is_header,
                    fill=fill,
                    line_color="D9E2E2",
                    align="ctr",
                )
                cur_x += col_widths[c]

    def xml(self):
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            "<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val=\"FFFFFF\"/>"
            "</a:solidFill></p:bgPr></p:bg>"
            "<p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/>"
            "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>"
            "<a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
            "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/>"
            "</a:xfrm></p:grpSpPr>"
            + "".join(self.items)
            + "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>"
            "</p:clrMapOvr></p:sld>"
        )


def slide1():
    s = SlideBuilder("EDMにPNDMサンプラーを移植した結果と分析")
    s.textbox(
        0.65,
        1.1,
        12.0,
        0.55,
        ["条件：CIFAR-10 conditional pretrained / 50,000 samples / 18 steps / 3 seed ranges"],
        16,
        "333333",
        fill="EEF5F5",
        line_color="D9E2E2",
        radius=True,
    )
    s.textbox(0.65, 1.62, 5.5, 0.35, ["指標：FID（小さいほど良い）"], 15, "555555")
    rows = [
        ["サンプラー", "NFE", "平均FID", "EDMとの差"],
        ["EDM Heun", "35", "1.8391", "0.0000"],
        ["PNDM", "約27", "1.9915", "+0.1525"],
        ["S-PNDM", "約19", "3.1477", "+1.3086"],
    ]
    s.table(1.0, 2.1, [3.4, 1.7, 2.3, 2.3], 0.62, rows)
    return s


def slide2():
    s = SlideBuilder("分析：EDM設計とPNDM前提の違い")
    s.textbox(
        0.75,
        1.15,
        5.65,
        0.72,
        ["dx / d sigma = (x - D(x, sigma)) / sigma"],
        20,
        "0F3D46",
        True,
        fill="F3F7F7",
        line_color="B9D4D4",
        align="ctr",
        radius=True,
    )
    s.textbox(
        0.75,
        2.05,
        5.65,
        2.35,
        [
            "EDMは sigma を時間変数として扱う",
            "Heun法は sigma 空間上のODEと整合",
            "pretrainedモデルもEDM設定で訓練",
            "rho schedule・preconditioning・loss weightingと組み合わせて性能を出す",
        ],
        16,
        "222222",
        bullet=True,
    )
    s.textbox(
        6.95,
        1.15,
        5.45,
        0.72,
        ["4次 Adams-Bashforth"],
        20,
        "0F3D46",
        True,
        fill="F3F7F7",
        line_color="B9D4D4",
        align="ctr",
        radius=True,
    )
    s.textbox(
        6.95,
        2.05,
        5.45,
        2.35,
        [
            "PNDMはVP/DDIM系の離散時刻を前提",
            "AB4の標準形は等間隔ステップを前提",
            "今回はEDMのrho schedule上で適用",
            "不等間隔sigma上では4次近似の前提とずれる可能性",
        ],
        16,
        "222222",
        bullet=True,
    )
    s.textbox(
        1.0,
        5.05,
        11.0,
        0.85,
        [
            "数値解法の次数だけでなく、ODE設計・時間変数・step間隔との相性がFIDに影響する"
        ],
        18,
        "0F3D46",
        True,
        fill="E6F1F1",
        line_color="A9C9C9",
        align="ctr",
        radius=True,
    )
    return s


def slide3():
    s = SlideBuilder("今後の実験予定")
    s.textbox(
        0.8,
        1.2,
        5.6,
        0.55,
        ["1. EDMと同じNFEに揃える"],
        18,
        "0F3D46",
        True,
        fill="F3F7F7",
        line_color="B9D4D4",
        radius=True,
    )
    s.textbox(
        0.95,
        2.0,
        5.35,
        3.25,
        [
            "EDM Heun：18 stepsで35 NFE",
            "PNDM：18 stepsで約27 NFE",
            "PNDMのstepsを増やして約35 NFEに合わせる",
            "FIDだけでなく生成時間も比較",
        ],
        16,
        "222222",
        bullet=True,
    )
    s.textbox(
        6.95,
        1.2,
        5.55,
        0.55,
        ["2. 4次法・線形多段法を追加比較"],
        18,
        "0F3D46",
        True,
        fill="F3F7F7",
        line_color="B9D4D4",
        radius=True,
    )
    s.textbox(
        7.1,
        2.0,
        5.25,
        3.25,
        [
            "4次 Runge-Kutta法をEDM ODE上で評価",
            "線形多段法を複数パターンで比較",
            "等間隔用AB4と可変刻み係数の違いを確認",
            "sigma scheduleとの相性を切り分ける",
        ],
        16,
        "222222",
        bullet=True,
    )
    return s


def rels_xml(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    for idx in range(slide_count):
        rels.append(
            f'<Relationship Id="rId{idx + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx + 1}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def presentation_xml(slide_count: int) -> str:
    ids = "".join(
        f'<p:sldId id="{256 + idx}" r:id="rId{idx + 3}"/>'
        for idx in range(slide_count)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f"<p:sldIdLst>{ids}</p:sldIdLst>"
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )


def content_types_xml(slide_count: int) -> str:
    slide_overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for idx in range(1, slide_count + 1)
    )
    return (
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
"""
        + slide_overrides
        + """
</Types>
"""
    )

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

def app_xml(slide_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>Codex</Application><PresentationFormat>ワイド画面</PresentationFormat><Slides>{slide_count}</Slides><Company></Company>
</Properties>
"""


def core_xml(title: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>{xml_escape(title)}</dc:title><dc:creator>Codex</dc:creator>
</cp:coreProperties>
"""

THEME_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Simple">
<a:themeElements><a:clrScheme name="Simple"><a:dk1><a:srgbClr val="222222"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="0F3D46"/></a:dk2><a:lt2><a:srgbClr val="EEF5F5"/></a:lt2><a:accent1><a:srgbClr val="0F3D46"/></a:accent1><a:accent2><a:srgbClr val="A9C9C9"/></a:accent2><a:accent3><a:srgbClr val="E6F1F1"/></a:accent3><a:accent4><a:srgbClr val="D9E2E2"/></a:accent4><a:accent5><a:srgbClr val="555555"/></a:accent5><a:accent6><a:srgbClr val="F4F7F7"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Yu Gothic"><a:majorFont><a:latin typeface="Yu Gothic"/><a:ea typeface="Yu Gothic"/></a:majorFont><a:minorFont><a:latin typeface="Yu Gothic"/><a:ea typeface="Yu Gothic"/></a:minorFont></a:fontScheme><a:fmtScheme name="Simple"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>
"""

SLIDE_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>
"""

SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""

SLIDE_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>
"""

SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def slide_rels_xml(media_entries) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for rel_id, _path, target in media_entries:
        rels.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def build_pptx_from_slides(path: str, slides: list[SlideBuilder], title: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as pptx:
        pptx.writestr("[Content_Types].xml", content_types_xml(len(slides)))
        pptx.writestr("_rels/.rels", ROOT_RELS)
        pptx.writestr("docProps/app.xml", app_xml(len(slides)))
        pptx.writestr("docProps/core.xml", core_xml(title))
        pptx.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        pptx.writestr("ppt/_rels/presentation.xml.rels", rels_xml(len(slides)))
        pptx.writestr("ppt/theme/theme1.xml", THEME_XML)
        pptx.writestr("ppt/slideMasters/slideMaster1.xml", SLIDE_MASTER)
        pptx.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", SLIDE_MASTER_RELS)
        pptx.writestr("ppt/slideLayouts/slideLayout1.xml", SLIDE_LAYOUT)
        pptx.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", SLIDE_LAYOUT_RELS)
        media_counter = 1
        for idx, slide in enumerate(slides, start=1):
            pptx.writestr(f"ppt/slides/slide{idx}.xml", slide.xml())
            media_entries = []
            for rel_id, media_path in slide.media:
                target = f"image{media_counter}{Path(media_path).suffix.lower()}"
                media_entries.append((rel_id, media_path, target))
                pptx.write(media_path, f"ppt/media/{target}")
                media_counter += 1
            pptx.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels_xml(media_entries))


def build_pptx(path: str) -> None:
    slides = [slide1(), slide2(), slide3()]
    build_pptx_from_slides(path, slides, "EDMにPNDMサンプラーを移植した結果と分析")


if __name__ == "__main__":
    build_pptx("experiments/reports/pndm_edm_analysis.pptx")
