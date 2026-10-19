#!/usr/bin/env python3
"""Build an editable seminar deck from the recorded Restart experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import posixpath
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from create_pndm_analysis_pptx import SlideBuilder, build_pptx_from_slides, emu, xml_escape

OUT = ROOT / "experiments/8/report/1003/restart_sampling_progress_report_codex.pptx"
EXP3 = ROOT / "experiments/8/3_restart_k5_fid50k_3seeds"
EXP4 = ROOT / "experiments/8/4_restart_k_sweep_10k_interval"
EXP6 = ROOT / "experiments/8/6_exact_10k_restart_k5_fid50k_1seed"
TRAIN = ROOT / "experiments/0/1_cifar10_cond_edm_10k_retraining"
RUN = TRAIN / "training-runs/cifar10-cond-ddpmpp-edm-seed0-10k"
FONT = "Noto Sans CJK JP"
INK, TEAL, MUTED = "202A30", "006D77", "5D666D"
ORANGE, GREEN, RULE = "B94C25", "00866C", "D8E0E3"
SVG_NS = {"s": "http://www.w3.org/2000/svg"}
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"


def load_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def signed(value):
    return f"{float(value):+.3f}"


class SeminarSlide(SlideBuilder):
    def __init__(self, title, number, section=""):
        self.shape_id, self.items, self.media = 2, [], []
        self.background = "FFFFFF"
        self.notes = ""
        self.text_bounds = []
        self.image_bounds = []
        if title:
            self.text(0.55, 0.30, 12.2, 0.60, title, 26, bold=True)
            self.line(0.55, 1.00, 12.23, RULE, 0.8)
            self.line(0.55, 1.00, 0.75, TEAL, 2.5)
        self.text(0.55, 7.10, 10.0, 0.23,
                  section or "CIFAR-10 class-conditional EDM", 10, MUTED)
        self.text(12.20, 7.05, 0.55, 0.30, str(number), 11, MUTED, align="r")

    def text(self, x, y, w, h, text, size=20, color=INK, bold=False,
             fill=None, align="l", valign="t"):
        sid = self.next_id()
        lines = text.split("\n")
        paragraphs = []
        for line in lines:
            paragraphs.append(
                f'<a:p><a:pPr algn="{align}"><a:lnSpc><a:spcPct val="115000"/>'
                '</a:lnSpc><a:buNone/></a:pPr><a:r>'
                f'<a:rPr lang="ja-JP" sz="{round(size * 100)}" b="{int(bold)}">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/>'
                '</a:rPr>'
                f'<a:t>{xml_escape(line)}</a:t></a:r>'
                f'<a:endParaRPr lang="ja-JP" sz="{round(size * 100)}"/></a:p>'
            )
        fill_xml = (f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
                    if fill else '<a:noFill/>')
        self.items.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'{fill_xml}<a:ln><a:noFill/></a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" anchor="{valign}" '
            'lIns="0" rIns="0" tIns="0" bIns="0"/><a:lstStyle/>'
            + ''.join(paragraphs) + '</p:txBody></p:sp>'
        )
        self.text_bounds.append((x, y, w, h, text, size, bold))

    def segment(self, x1, y1, x2, y2, color=TEAL, width=2.5, arrow=False):
        sid = self.next_id()
        flips = (' flipH="1"' if x2 < x1 else '') + (' flipV="1"' if y2 < y1 else '')
        end = '<a:tailEnd type="triangle" w="sm" len="sm"/>' if arrow else ''
        self.items.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="Path {sid}"/>'
            '<p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr>'
            f'<a:xfrm{flips}><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/>'
            f'<a:ext cx="{emu(abs(x2 - x1))}" cy="{emu(abs(y2 - y1))}"/></a:xfrm>'
            '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:ln w="{int(width * 12700)}"><a:solidFill>'
            f'<a:srgbClr val="{color}"/></a:solidFill>{end}</a:ln></p:spPr></p:cxnSp>'
        )

    def cropped_picture(self, x, y, w, h, path, bounds):
        with Image.open(path) as source:
            iw, ih = source.size
        left, top, right, bottom = bounds
        if not (0 <= left < right <= iw and 0 <= top < bottom <= ih):
            raise ValueError(f"Invalid image crop: {bounds} for {path}")
        original_aspect = (right - left) / (bottom - top)
        if abs(original_aspect / (w / h) - 1) > 0.001:
            raise ValueError(f"Image crop would be distorted: {path}")
        sid = self.next_id()
        rid = f"rId{len(self.media) + 2}"
        self.media.append((rid, str(path)))
        crop = dict(l=round(100000 * left / iw), t=round(100000 * top / ih),
                    r=round(100000 * (iw - right) / iw),
                    b=round(100000 * (ih - bottom) / ih))
        attrs = ' '.join(f'{key}="{value}"' for key, value in crop.items())
        self.items.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="Result figure {sid}"/>'
            '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
            '<p:nvPr/></p:nvPicPr><p:blipFill>'
            f'<a:blip r:embed="{rid}"/><a:srcRect {attrs}/>'
            '<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>'
            f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '<a:ln><a:noFill/></a:ln></p:spPr></p:pic>'
        )
        self.image_bounds.append((x, y, w, h))

    def takeaway(self, text, size=20):
        self.line(0.55, 6.57, 12.23, RULE, 0.8)
        self.text(0.65, 6.69, 12.0, 0.36, text, size, TEAL, True)


def svg_axes_bounds(path, bottom_margin=37):
    root = ET.parse(path).getroot()
    _, _, sw, sh = map(float, root.attrib["viewBox"].split())
    with Image.open(path.with_suffix(".png")) as image:
        pw, ph = image.size
    bounds = []
    for group in root.findall('.//s:g', SVG_NS):
        if not re.fullmatch(r'axes_\d+', group.get('id', '')):
            continue
        patch = group.find('s:g/s:path', SVG_NS)
        if patch is None:
            continue
        values = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', patch.get('d', ''))]
        xs, ys = values[::2], values[1::2]
        # Include the axes, ticks, titles, and labels, but not the overall experiment title.
        left = max(0, min(xs) - 55)
        right = min(sw, max(xs) + 8)
        top = max(0, min(ys) - 24)
        bottom = min(sh, max(ys) + bottom_margin)
        bounds.append((left * pw / sw, top * ph / sh,
                       right * pw / sw, bottom * ph / sh))
    return bounds


def overview(slide, path, y=2.02, max_h=4.2):
    axes = svg_axes_bounds(path.with_suffix('.svg'), bottom_margin=47)
    with Image.open(path) as image:
        width, height = image.size
    bounds = (0, min(b[1] for b in axes), width, max(b[3] for b in axes))
    h = 12.23 * (bounds[3] - bounds[1]) / width
    w = 12.23
    if h > max_h:
        w *= max_h / h
        h = max_h
    slide.cropped_picture((13.3333 - w) / 2, y, w, h, path, bounds)
    return y + h


def training_status():
    snapshots = sorted(int(p.stem.split('-')[-1]) for p in RUN.glob('network-snapshot-*.pkl')
                       if p.stat().st_size and (RUN / p.name.replace('network-snapshot-',
                           'training-state-').replace('.pkl', '.pt')).is_file())
    with (TRAIN / 'logs/train.log').open(encoding='utf-8', errors='replace') as stream:
        kimgs = [float(match.group(1)) for line in stream
                 if (match := re.search(r'\bkimg\s+(\d+(?:\.\d+)?)', line))]
    return snapshots, kimgs[-1]


def make_slides():
    stats = load_tsv(EXP3 / 'results/restart_k5_fid50k_3seed_stats.tsv')
    optimal = load_tsv(EXP4 / 'results/optimal_k.tsv')
    progressive = load_tsv(EXP6 / 'results/restart_improvements.tsv')
    paired25 = [r for r in progressive if int(r['num_images']) == 25000]
    sweep25 = load_tsv(EXP4 / 'results/fid25k/restart_fid_summary.tsv')
    checkpoints, train_kimg = training_status()
    assert len(stats) == 4 and all(r['complete'] == '1' and r['n_seed_ranges'] == '3' for r in stats)
    assert len(optimal) == 10 and len(paired25) == 10
    first, last = paired25[0], paired25[-1]
    saved = checkpoints[-1] // 1000
    common_notes = (
        '共通条件：CIFAR-10 class-conditional EDM。main samplerはEDM Heun、'
        '18 steps、rho=7、S_churn=0。Restart設定はN_restart=3、'
        't_min=0.06、t_max=0.30、noise multiplier=1。'
        't_minはmain scheduleの最も近い点に合わせる。NFE=35+4K。'
        'FID referenceはfid-refs/cifar10-32x32.npz。'
        '改善幅=FID(ODE)-FID(Restart)。正値ならRestartによる改善。'
    )
    slides = []
    s = SeminarSlide('', 1)
    s.line(0.85, 2.13, 0.8, TEAL, 4)
    s.text(0.85, 2.50, 11.6, 0.9, '研究進捗', 48, bold=True)
    s.text(0.85, 3.76, 11.65, 1.12,
           'CIFAR-10 class-conditional EDMにおける\n再学習、Restart実験', 25)
    s.text(0.90, 5.73, 11.5, 0.42,
           '再学習  /  最適なRestart回数の探索  /  効果の再確認', 18, MUTED)
    s.notes = '今回はCIFAR-10 class-conditional EDMの再学習とRestart Samplingの進捗を報告する。日付は表示しない。'
    slides.append(s)

    s = SeminarSlide('Restart Samplingの仕組みと比較指標', 2)
    s.text(0.6, 1.24, 12.1, 0.62,
           '生成途中でノイズを加え、同じモデルでもう一度ノイズを除去する。', 21)
    s.text(0.65, 2.00, 7.8, 0.35, '通常ODE：ノイズを段階的に減らす', 18, bold=True)
    s.text(1.15, 2.55, 1.15, 0.28, 'ノイズ大', 13, MUTED)
    for a, b in zip([(2.25, 2.6), (4.6, 3.02), (6.5, 3.24)],
                    [(4.6, 3.02), (6.5, 3.24), (8.25, 3.35)]):
        s.segment(*a, *b, MUTED, arrow=True)
    s.text(7.9, 3.49, 0.7, 0.28, '画像', 13, MUTED)
    s.text(0.65, 3.97, 7.8, 0.35, 'Restart：途中の区間でノイズ追加と除去を繰り返す', 18, bold=True)
    for a, b, color in [((2.25, 4.65), (4.5, 5.05), MUTED),
                         ((4.5, 5.05), (5.45, 4.49), ORANGE),
                         ((5.45, 4.49), (6.55, 5.05), TEAL),
                         ((6.55, 5.05), (8.25, 5.21), MUTED)]:
        s.segment(*a, *b, color, arrow=True)
    s.text(4.35, 4.37, 1.3, 0.30, 'ノイズ追加', 13, ORANGE)
    s.text(5.80, 4.35, 1.2, 0.30, '再び除去', 13, TEAL)
    s.text(4.47, 5.36, 3.45, 0.35, 'この区間をK回繰り返す', 15, TEAL, True)
    s.text(7.9, 5.4, 0.7, 0.28, '画像', 13, MUTED)
    s.segment(8.86, 2.04, 8.86, 5.87, RULE, 1)
    for y, label, detail in [(2.03, 'K：Restartの回数', 'K=0はRestartなし'),
                              (3.23, 'NFE：モデル呼び出し回数', '画像1枚あたりの計算量の目安'),
                              (4.43, 'FID：生成画像の評価指標', '低いほど良い')]:
        s.text(9.16, y, 3.65, 0.44, label, 17, bold=True)
        s.text(9.16, y + 0.48, 3.65, 0.44, detail, 15, MUTED)
    s.text(0.65, 6.07, 12.0, 0.37,
           '今回の設定：通常ODEは35 NFE、Restartは35+4K NFE（K=5で55 NFE）。', 17)
    s.takeaway('計算量が増えるため、同じNFEのODEとも比較する。', 20)
    s.notes = common_notes + '模式図は仕組みの説明であり、実測のノイズ軌道ではない。'
    slides.append(s)

    s = SeminarSlide('既存モデルでのRestart効果の検証', 3, '既存モデル / FID-50k / 生成seed範囲3組')
    s.text(0.60, 1.23, 12.0, 0.35, '目的：学習初期と後期でRestartの効果が変わるか確認する。', 18)
    s.text(0.60, 1.68, 12.0, 0.35,
           '条件：5.5k・20k・120k・200k、K=5、50,000枚 × 独立した生成seed範囲3組', 17)
    figure = EXP3 / 'results/restart_k5_fid50k_3seed_overview.png'
    end = overview(s, figure, y=2.10, max_h=3.98)
    s.text(0.65, end + 0.08, 12.0, 0.34,
           '左：平均FID（低いほど良い）　右：FID改善幅（正値が改善）　誤差棒：標本標準偏差', 13)
    s.takeaway('学習初期では改善したが、学習後期では悪化した。')
    s.notes = common_notes + (
        'checkpointは既存系列の5.5k、20k、120k、200k。通常ODEは18 steps/35 NFE、'
        '同一計算量ODEは28 steps/55 NFE。独立した生成seed範囲は0-49999、'
        '50000-99999、100000-149999。各範囲50,000枚。学習の3回再現実験ではない。'
        '誤差棒はseed範囲間の標本標準偏差であり信頼区間ではない。'
        '同一NFE改善幅の平均は' + '、'.join(signed(r['improvement_vs_same_nfe_mean']) for r in stats)
        + '。出典：' + str(figure.relative_to(ROOT))
    )
    slides.append(s)

    s = SeminarSlide('学習段階を細かく比較するための再学習', 4)
    s.text(0.65, 1.39, 1.25, 0.46, '目的', 18, TEAL, True)
    s.text(2.00, 1.36, 10.55, 1.02,
           'これまで評価したcheckpointの間隔が不均一だったため、\n厳密な10k間隔のcheckpoint系列を作り直す。', 23, bold=True)
    s.line(0.65, 2.74, 11.95, RULE, 0.8)
    s.text(0.65, 3.01, 1.25, 0.46, '学習条件', 18, TEAL, True)
    s.text(2.00, 3.00, 10.55, 1.52,
           'CIFAR-10 class-conditional / DDPM++ / EDM\n学習率 0.001、EMA 0.5 Mimg、学習seed=0\n200kまで連続学習し、10kごとに保存', 21)
    s.line(0.65, 4.90, 11.95, RULE, 0.8)
    s.text(0.65, 5.16, 1.25, 0.46, '進捗', 18, TEAL, True)
    s.text(2.00, 5.12, 10.55, 0.57,
           f'約{train_kimg / 1000:.0f}kまで到達 / {saved}kまで保存済み', 25, TEAL, True)
    s.text(2.00, 5.91, 10.55, 0.56,
           '10k = 学習で処理した1,000万枚（10,000 kimg）\n※ optimizerの更新回数ではない。', 14, MUTED)
    s.notes = (
        '出典：experiments/0/1_cifar10_cond_edm_10k_retraining/README.md、logs/train.log、training-runs。'
        f'生成時点の最終log kimg={train_kimg}。保存済みsnapshot/state対は{len(checkpoints)}点。'
        'バッチ512、batch-gpu128、Adam、dropout0.13、augment0.12、FP32、1 GPU。'
        '10k間隔は保存ラベルであり、指定点以上になる最初のoptimizer update直後に保存する。'
        'バッチによる最大超過は0.384 kimg。学習の動作中状態は断定せず到達logを報告。'
        'このページに図表は掲載しない。'
    )
    slides.append(s)

    panel_source = EXP4 / 'results/fid_vs_k_by_checkpoint.png'
    crops = svg_axes_bounds(panel_source.with_suffix('.svg'))
    assert len(crops) == len(optimal)
    for page, (indices, tag, takeaway) in enumerate([
        (range(5), '10k〜50k', '最良Kはcheckpointによって異なり、学習初期ほど改善幅が大きい。'),
        (range(5, 10), '60k〜100k', '後半の改善幅は小さい。最良Kは25,000枚評価でも確認する。')
    ], 5):
        s = SeminarSlide('学習段階ごとの最適なRestart回数の探索', page, '再学習系列 / FID-5k / K=0〜8')
        s.text(0.60, 1.18, 12.0, 0.35,
               f'目的：学習段階ごとの最良Kを調べる。　対象：{tag}', 17)
        s.text(0.60, 1.63, 12.0, 0.33,
               '条件：K=0〜8、各条件5,000枚、NFE=35+4K　｜　破線：K=0、星印：最小FID', 15)
        panel_w = 3.15
        positions = [(1.64, 2.05), (5.09, 2.05), (8.54, 2.05),
                     (3.36, 4.20), (6.81, 4.20)]
        for i, (x, y) in zip(indices, positions):
            bounds = crops[i]
            height = panel_w * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
            # Preserve source axis/tick geometry, including the minimum-FID annotation.
            s.cropped_picture(x, y, panel_w, height, panel_source, bounds)
        s.text(0.65, 6.37, 12.0, 0.21,
               '各パネルで縦軸の範囲は異なる。5,000枚評価は候補選択用。', 11, MUTED)
        s.takeaway(takeaway, 18)
        selected = [optimal[i] for i in indices]
        s.notes = common_notes + (
            '10k刻み再学習系列。各条件5,000枚、seeds0-4999。'
            '最良KとK=0からの改善幅：'
            + '、'.join(f"{r['checkpoint']} K={r['optimal_k']} 改善{float(r['improvement_vs_k0']):.5f}" for r in selected)
            + '。最良KはFID最小化基準でありNFE一定の最適化ではない。'
            'checkpointごとに縦軸範囲が違うため曲線の傾きだけで改善幅を比較しない。'
            '特に100kのK=1とK=2のFID差は0.00011であり、最適Kを断定しない。'
            '出典：' + str(panel_source.relative_to(ROOT))
        )
        slides.append(s)

    s = SeminarSlide('均一なチェックポイントでRestart効果の再確認', 7,
                     '再学習系列 / FID-25k / K=5')
    s.text(0.60, 1.20, 12.0, 0.35,
           '目的：10k間隔の系列でも、学習に伴ってRestartの効果が小さくなるか確認する。', 17)
    s.text(0.60, 1.65, 12.0, 0.35,
           '条件：10k〜100k、25,000枚、K=5（55 NFE）、通常ODE（35 NFE）・同一NFEのODEと比較', 16)
    figure = EXP6 / 'results/restart_k5_fid25000_overview.png'
    end = overview(s, figure, y=2.06, max_h=3.99)
    s.text(0.65, end + 0.10, 12.0, 0.35,
           '左：FIDの推移　右：ODEに対するFID改善幅（正値が改善）', 13, MUTED)
    s.text(0.65, 5.89, 12.0, 0.45,
           f"同一NFEでの改善幅：{first['checkpoint']}で{float(first['improvement_vs_same_nfe']):.3f} → "
           f"{last['checkpoint']}で{float(last['improvement_vs_same_nfe']):.3f}", 21, TEAL, True)
    s.takeaway('均一なcheckpoint系列でも、学習が進むとRestartの改善幅は小さくなった。', 18)
    s.notes = common_notes + (
        '再学習系列。seeds0-24999、25,000枚。通常ODE18 steps/35 NFE、'
        '同一計算量ODE28 steps/55 NFE。3条件が揃った10k-100kの10点のみを主図に掲載。'
        '110kは通常ODEとRestartのみの部分結果であり同一NFE比較は未完了。'
        'FID-5kは10k-120kで完了。FID-50kは未実施。'
        '旧系列のFID-50kと再学習系列のFID-25kはモデル系列と画像数が異なるため'
        'FID絶対値を直接比較しない。出典：' + str(figure.relative_to(ROOT))
    )
    slides.append(s)

    s = SeminarSlide('まとめと今後の確認', 8)
    s.text(0.65, 1.34, 12.0, 0.46, 'ここまでの結果', 20, TEAL, True)
    s.text(0.65, 2.03, 12.0, 0.60,
           '既存モデル：Restartは学習初期に改善し、後期には悪化した。', 22, bold=True)
    s.text(0.65, 2.82, 12.0, 0.66,
           '再学習系列：学習が進むにつれて、Restartの改善幅が小さくなった。', 21)
    s.line(0.65, 3.72, 11.95, RULE, 0.8)
    s.text(0.65, 4.04, 12.0, 0.46, '現在の確認', 20, TEAL, True)
    checkpoint25 = sorted({r['checkpoint'] for r in sweep25})
    description = '、'.join(
        f"{cp}のK=" + '・'.join(str(k) for k in sorted({int(r['restart_k']) for r in sweep25 if r['checkpoint'] == cp}))
        for cp in checkpoint25
    )
    s.text(0.65, 4.72, 12.0, 0.62,
           '最適Kの確認に向け、K探索を25,000枚評価へ拡張。', 22, bold=True)
    s.text(0.65, 5.52, 12.0, 0.58, f'評価済み：{description}。残りのKは確認途中。', 17)
    best25 = min(sweep25, key=lambda r: float(r['fid']))
    s.text(0.65, 6.04, 12.0, 0.35,
           f"暫定最良：{best25['checkpoint']}のK={best25['restart_k']}、FID={float(best25['fid']):.4f}（全K評価前）", 15, MUTED)
    s.notes = common_notes + (
        '旧系列の50,000枚×3 seed範囲では初期改善・後期悪化を確認。'
        '新系列は同じ学習設定だが別学習系列であり、K=5の25,000枚結果は100kまでの改善縮小。'
        'K探索25,000枚評価は記録済み結果のみ表示する。'
        '80kの暫定最良を最適Kとして断定しない。'
        'K=5の25,000枚評価は110kの同一NFE ODEと120kの全条件が未集計。'
        '再学習系列の50,000枚評価は未実施。'
        '出典：experiments/8/4_restart_k_sweep_10k_interval/results/fid25k/restart_fid_summary.tsv。'
    )
    slides.append(s)
    return slides


def add_notes(path, slides):
    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    types = ET.fromstring(files['[Content_Types].xml'])
    for i, slide in enumerate(slides, 1):
        name = f'ppt/notesSlides/notesSlide{i}.xml'
        ET.SubElement(types, f'{{{CT}}}Override', PartName='/' + name,
                      ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')
        paragraphs = ''.join(f'<a:p><a:r><a:rPr lang="ja-JP"/><a:t>{xml_escape(line)}</a:t></a:r></a:p>'
                             for line in slide.notes.split('\n'))
        files[name] = (
            f'<p:notes xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Speaker notes"/><p:cNvSpPr txBox="1"/>'
            '<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/>'
            '<p:txBody><a:bodyPr/><a:lstStyle/>' + paragraphs + '</p:txBody></p:sp>'
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'
        ).encode('utf-8')
        files[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels'] = (
            f'<Relationships xmlns="{REL}"><Relationship Id="rId1" '
            f'Type="{R}/slide" Target="../slides/slide{i}.xml"/></Relationships>'
        ).encode('utf-8')
        rel_path = f'ppt/slides/_rels/slide{i}.xml.rels'
        rels = ET.fromstring(files[rel_path])
        ET.SubElement(rels, f'{{{REL}}}Relationship', Id='rIdNotes',
                      Type=R + '/notesSlide', Target=f'../notesSlides/notesSlide{i}.xml')
        ET.register_namespace('', REL)
        files[rel_path] = ET.tostring(rels, encoding='utf-8', xml_declaration=True)
    ET.register_namespace('', CT)
    files['[Content_Types].xml'] = ET.tostring(types, encoding='utf-8', xml_declaration=True)
    files['ppt/theme/theme1.xml'] = files['ppt/theme/theme1.xml'].replace(b'Yu Gothic', FONT.encode())
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pptx', delete=False) as temp:
        temporary = Path(temp.name)
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as target:
        for name, content in files.items():
            target.writestr(name, content)
    temporary.replace(path)


def validate(slides, path):
    issues = []
    for i, slide in enumerate(slides, 1):
        regions = slide.image_bounds + [tuple(b[:4]) for b in slide.text_bounds]
        for index, (x, y, w, h) in enumerate(slide.image_bounds):
            for other in regions[index + 1:]:
                xx, yy, ww, hh = other
                if min(x + w, xx + ww) - max(x, xx) > 0.01 and min(y + h, yy + hh) - max(y, yy) > 0.01:
                    issues.append(f'Slide {i}: figure overlaps another figure or text')
        for x, y, w, h, text, size, bold in slide.text_bounds:
            font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-' + ('Bold' if bold else 'Regular') + '.ttc'
            font = ImageFont.truetype(font_path, round(size * 4))
            for line in text.split('\n'):
                needed = font.getlength(line) / 4 / 72
                if needed > w:
                    issues.append(f'Slide {i} text too wide ({needed:.2f}>{w:.2f}): {line}')
            needed_h = len(text.split('\n')) * size * 1.15 / 72
            if needed_h > h + 0.05:
                issues.append(f'Slide {i} text too tall ({needed_h:.2f}>{h:.2f}): {text}')
            if x < 0 or y < 0 or x + w > 13.334 or y + h > 7.5:
                issues.append(f'Slide {i} text outside slide: {text}')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            issues.append('PPTX zip CRC failed')
        for name in archive.namelist():
            if name.endswith(('.xml', '.rels')):
                ET.fromstring(archive.read(name))
            if name.endswith('.rels'):
                directory = posixpath.dirname(posixpath.dirname(name))
                for relationship in ET.fromstring(archive.read(name)):
                    if relationship.get('TargetMode') == 'External':
                        continue
                    target = relationship.get('Target', '')
                    resolved = posixpath.normpath(posixpath.join(directory, target))
                    if resolved not in archive.namelist():
                        issues.append(f'Missing relationship target: {name} -> {resolved}')
        visible = '\n'.join(
            ''.join(ET.fromstring(archive.read(f'ppt/slides/slide{i}.xml')).itertext())
            for i in range(1, len(slides) + 1)
        )
        if re.search(r'Experiment|実験[08]-|2026[-/]', visible):
            issues.append('Unwanted date/experiment identifier visible')
    if issues:
        raise ValueError('\n'.join(issues))
    print(json.dumps({'output': str(path), 'slides': len(slides),
                      'notes': len(slides), 'text_and_xml_checks': 'passed'}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    slides = make_slides()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_pptx_from_slides(str(args.output), slides, '研究進捗')
    add_notes(args.output, slides)
    validate(slides, args.output)


if __name__ == '__main__':
    main()
