// Build the background deck for new M1 students: from diffusion models to Restart Sampling.
//
// Usage (from the EDM repository root, after scripts/prepare_figures.py):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/background/scripts/build_report.js
//
// Paper figures are cited on each slide; the numbers quoted were checked against the papers:
// DDPM (arXiv:2006.11239), Score SDE (2011.13456), EDM (2206.00364), Restart (2306.14878).

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report");
const ASSETS = path.join(REPORT_DIR, "background", "assets");
const OUT = path.join(REPORT_DIR, "restart_background_claude.pptx");

const FONT = "Hiragino Sans";
const THEME = {
	name: "Restart Progress",
	headFontFace: FONT,
	bodyFontFace: FONT,
	colors: {
		dk1: "212529", lt1: "FFFFFF", dk2: "1A365D", lt2: "EEF2F6",
		accent1: "1A365D", accent2: "D55E00", accent3: "0072B2",
		accent4: "009E73", accent5: "5B7896", accent6: "9B4A4A",
		hlink: "0072B2", folHlink: "5B7896",
	},
};
const MUTED = "6C757D";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "研究の背景：拡散モデルからRestart Samplingへ";
const C = pres.SchemeColor;
const ORANGE = C.accent2, BLUE = C.accent3, GREEN = C.accent4, NAVY = C.text2, TINT = C.background2, SLATE = C.accent5;

const W = 13.333, MX = 0.55, CW = W - 2 * MX;

// ---------------------------------------------------------------------------- layouts

pres.defineSlideMaster({
	title: "TITLE",
	background: { color: THEME.colors.dk2 },
	objects: [
		{ placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.0, w: 11.5, h: 1.5, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
		{ placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.7, w: 11.5, h: 0.8, fontSize: 20, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
	],
});

pres.defineSlideMaster({
	title: "CONTENT",
	background: { color: THEME.colors.lt1 },
	objects: [
		{ placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: CW, h: 0.7, fontSize: 26, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0 }, text: "" } },
	],
	slideNumber: { x: 12.33, y: 7.02, w: 0.45, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

// ---------------------------------------------------------------------------- helpers

function text(slide, value, opts) {
	slide.addText(value, Object.assign({ isTextBox: true, margin: 0, valign: "top", color: C.text1, fontSize: 14, fit: "none" }, opts));
}

function card(slide, x, y, w, h, fill, name) {
	slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || TINT }, line: { type: "none" }, objectName: name });
}

function chip(slide, label, x, y, w, fill, textColor) {
	slide.addText(label, {
		shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.15, x, y, w, h: 0.3,
		fill: { color: fill || NAVY }, line: { type: "none" }, color: textColor || C.background1,
		fontSize: 11.5, bold: true, align: "center", valign: "middle", margin: 0, objectName: `chip ${label}`,
	});
}

function picture(slide, file, x, y, w, name) {
	const { width, height } = pngSize(path.join(ASSETS, file));
	const h = w * height / width;
	slide.addImage({ path: path.join(ASSETS, file), x, y, w, h, objectName: name });
	return h;
}

// Width and height from the PNG header.
function pngSize(file) {
	const buf = fs.readFileSync(file);
	return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

function source(slide, value, y) {
	text(slide, `出典：${value}`, { x: MX, y: y || 6.92, w: 10.5, h: 0.25, fontSize: 10, color: MUTED });
}

// Straight segment from (x1, y1) to (x2, y2); the arrow head, if any, sits at (x2, y2).
function seg(slide, x1, y1, x2, y2, color, width, extra) {
	const opts = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: Object.assign({ color, width }, extra || {}) };
	if ((x2 < x1) !== (y2 < y1)) opts.flipV = true;
	if (x2 < x1) opts.flipH = !opts.flipH;
	slide.addShape(pres.shapes.LINE, opts);
}

const sub = (t) => ({ text: t, options: { subscript: true } });

// ---------------------------------------------------------------------------- 1. title

pres.addSection({ title: "導入" });
{
	const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "導入" });
	s.addText("研究の背景\n拡散モデルからRestart Samplingへ", { placeholder: "title" });
	s.addText("進捗報告を理解するための前提知識", { placeholder: "body" });
	// Faint Restart trajectory as the cover motif: denoise, then noise-up / re-denoise loops.
	const top = 5.15, tmax = 5.95, tmin = 6.35, bottom = 6.75;
	seg(s, 7.0, top, 9.6, tmin, C.background2, 2.25, { transparency: 55 });
	for (let i = 0; i < 3; i++) {
		const x = 9.6 + i * 0.6;
		seg(s, x, tmin, x, tmax, ORANGE, 2.25, { transparency: 25 });
		seg(s, x, tmax, x + 0.6, tmin, GREEN, 2.25, { transparency: 25 });
	}
	seg(s, 11.4, tmin, 12.4, bottom, C.background2, 2.25, { transparency: 55 });
	s.addNotes("進捗報告に入る前に、M1の方にも実験の中身が伝わるよう、前提となる拡散モデルの流れを10分弱で説明します。最初の手法であるDDPMから、本研究の土台であるEDM、そして今回扱っているRestart Samplingまでを、「なぜ次の手法が必要になったか」という流れでつなげます。");
}

// ---------------------------------------------------------------------------- 2. what is a diffusion model

pres.addSection({ title: "背景" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景" });
	s.addText("拡散モデルとは", { placeholder: "title" });

	const rows = [
		["加工（学習時）", ORANGE, "画像にガウスノイズを少しずつ加え、最後は純粋なノイズにする"],
		["生成", BLUE, "純粋なノイズから出発し、ネットワークが少しずつノイズを取り除いて画像にする"],
	];
	rows.forEach(([label, color, body], i) => {
		const y = 1.3 + i * 0.62;
		chip(s, label, MX, y, 1.6, color);
		text(s, body, { x: MX + 1.8, y: y + 0.01, w: 6.0, h: 0.5, fontSize: 15 });
	});

	card(s, 8.6, 1.2, W - MX - 8.6, 1.95, TINT, "formula");
	text(s, "ノイズの加え方（本研究での書き方）", { x: 8.85, y: 1.33, w: 3.8, h: 0.3, fontSize: 12, bold: true, color: NAVY });
	text(s, [{ text: "x" }, sub("σ"), { text: " = x" }, sub("0"), { text: " + σ ε" }], { x: 8.85, y: 1.7, w: 3.8, h: 0.5, fontSize: 24, bold: true, color: NAVY, align: "center" });
	text(s, "ε：標準正規ノイズ　σ：ノイズの大きさ", { x: 8.85, y: 2.35, w: 3.8, h: 0.3, fontSize: 12, color: MUTED, align: "center" });
	text(s, [
		{ text: "ネットワークが学ぶのは ", options: { breakLine: false } },
		{ text: "デノイザ D(x; σ)", options: { bold: true, color: NAVY, breakLine: true } },
		{ text: "＝ ノイズの乗った画像から、元の画像を推定する関数" },
	], { x: MX, y: 2.6, w: 7.8, h: 0.7, fontSize: 15, paraSpaceAfter: 2 });

	const fw = 10;
	const fh = picture(s, "ddpm_fig2.png", (W - fw) / 2, 3.6, fw, "figure DDPM Fig 2");
	text(s, "x_T（純粋なノイズ）から x_0（画像）へ。右向き p_θ：ネットワークによるノイズ除去（生成）、左向きの破線 q：ノイズの付加（加工）",
		{ x: MX, y: 3.6 + fh + 0.15, w: CW, h: 0.3, fontSize: 12, color: MUTED, align: "center" });
	source(s, "Ho et al., 2020（DDPM）, Fig. 2");

	s.addNotes("拡散モデルは、画像にノイズを少しずつ加えて最後は純粋なノイズにする加工過程と、その逆に、ノイズから少しずつノイズを取り除いて画像を作る生成過程からなります。下の図の左端のx_Tが純粋なノイズ、右端のx_0が画像です。左向きの破線がノイズを加える加工、右向きの矢印がノイズを取り除く生成にあたります。ネットワークが学習するのは、ノイズの乗った画像から元の画像を推定する関数で、デノイザと呼びます。生成時は、このデノイザを何度も呼び出して、ノイズを少しずつ取り除いていきます。");
}

// ---------------------------------------------------------------------------- 3. DDPM

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景" });
	s.addText("DDPM（Ho et al., 2020）", { placeholder: "title" });

	const lw = 5.6;
	const points = [
		["生成", "ノイズを取り除く処理を T = 1000 ステップ繰り返す"],
		["品質", "CIFAR-10でFID 3.17。当時のGANに匹敵する画質"],
	];
	points.forEach(([label, body], i) => {
		const y = 1.4 + i * 0.95;
		chip(s, label, MX, y, 0.8);
		text(s, body, { x: MX + 1.0, y: y + 0.01, w: lw - 1.0, h: 0.7, fontSize: 15 });
	});
	card(s, MX, 3.3, lw, 1.95, TINT, "problem");
	chip(s, "課題", MX + 0.25, 3.47, 0.8, ORANGE);
	text(s, [
		{ text: "1枚の生成にネットワークを1000回呼ぶ必要があり、遅い", options: { bold: true, breakLine: true } },
		{ text: "→ 品質を保ったまま、少ないステップで生成したい", options: { color: NAVY } },
	], { x: MX + 0.25, y: 3.9, w: lw - 0.5, h: 1.25, fontSize: 15, paraSpaceAfter: 4 });

	const rx = MX + lw + 0.4, rw = W - MX - rx;
	const fh = picture(s, "ddpm_fig1.png", rx, 1.35, rw, "figure DDPM Fig 1");
	text(s, "DDPMの生成画像（左：CelebA-HQ 256×256、右：CIFAR-10）", { x: rx, y: 1.35 + fh + 0.08, w: rw, h: 0.3, fontSize: 11, color: MUTED });
	source(s, "Ho et al., 2020（DDPM）, Fig. 1");

	s.addNotes("拡散モデルが注目されるきっかけになったのが、2020年のDDPMです。前のページの図は、このDDPMの論文のものです。右の画像のように、CIFAR-10でFID 3.17と、当時のGANに匹敵する画質を出しました。一方で、ノイズを取り除くステップを1000回繰り返すので、1枚作るのにネットワークを1000回呼ぶ必要があり、とても遅いという課題がありました。ここから、品質を保ったまま少ないステップで生成する方法が研究されていきます。");
}

// ---------------------------------------------------------------------------- 4. SDE and ODE

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景" });
	s.addText("SDEとODE（Song et al., 2021）", { placeholder: "title" });
	text(s, "同じ学習済みモデルを使って、生成の進め方が2通りある", { x: MX, y: 1.05, w: CW, h: 0.35, fontSize: 16, color: SLATE });

	const lw = 6.9;
	const fh = picture(s, "score_sde_fig2.png", MX, 1.6, lw, "figure Score SDE Fig 2");
	text(s, "赤系の線：SDEの軌跡（ランダムに揺れる）　白い線：ODEの軌跡（なめらか）", { x: MX, y: 1.6 + fh + 0.06, w: lw, h: 0.3, fontSize: 11, color: MUTED });

	card(s, MX, 4.6, lw, 1.45, TINT, "formula");
	text(s, "確率フローODE（EDMの書き方）", { x: MX + 0.25, y: 4.72, w: 4, h: 0.3, fontSize: 12, bold: true, color: NAVY });
	text(s, "dx / dt = ( x − D(x; t) ) / t", { x: MX + 0.25, y: 5.05, w: lw - 0.5, h: 0.45, fontSize: 22, bold: true, color: NAVY, align: "center" });
	text(s, "デノイザ D の出力に向かって、少しずつ x を動かす", { x: MX + 0.25, y: 5.6, w: lw - 0.5, h: 0.3, fontSize: 12, color: MUTED, align: "center" });

	const rx = MX + lw + 0.35, rw = W - MX - rx;
	const items = [
		["逆SDE（確率的）", ORANGE, "ノイズを除きながら、毎ステップ少しノイズを加える", "誤差を打ち消せるが、細かい刻み（多くのステップ）が必要"],
		["確率フローODE（決定論的）", BLUE, "ノイズを加えずに、決まった道筋で除く", "大きな刻みで速く解けるが、誤差が残りやすい"],
	];
	items.forEach(([label, color, how, prosCons], i) => {
		const y = 1.6 + i * 1.7;
		card(s, rx, y, rw, 1.5, TINT, label);
		chip(s, label, rx + 0.2, y + 0.17, 2.6, color);
		text(s, [{ text: how, options: { breakLine: true } }, { text: prosCons, options: { bold: true, color: NAVY } }], { x: rx + 0.2, y: y + 0.6, w: rw - 0.4, h: 0.85, fontSize: 13.5, paraSpaceAfter: 4 });
	});
	text(s, "ODEを少ないステップで解く工夫として、DDIMやDPM-Solverなど（これまでのゼミで紹介）がある", { x: rx, y: 5.08, w: rw, h: 0.9, fontSize: 13, color: SLATE });
	source(s, "Song et al., 2021（Score SDE）, Fig. 2");

	s.addNotes("2021年に、拡散モデルを確率微分方程式、SDEの枠組みで整理した研究が出ました。ここで分かったのは、同じ学習済みモデルを使って、生成の進め方が2通りあるということです。1つは逆SDEで、ノイズを取り除きながら毎ステップ少しノイズを加えます。図の赤い線のようにランダムに揺れますが、途中の誤差を打ち消す働きがあります。ただし細かい刻みが必要で遅いです。もう1つは確率フローODEで、ノイズを加えずに白い線のような決まった道筋をたどります。大きな刻みで速く解けますが、誤差が残りやすいです。DDIMやDPM-Solverは、このODEを少ないステップで解く工夫です。");
}

// ---------------------------------------------------------------------------- 5. EDM

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景" });
	s.addText("EDM（Karras et al., 2022）", { placeholder: "title" });
	text(s, "設計要素（ノイズの付け方・スケジュール・前処理・ソルバー）を1つの枠組みで整理し、それぞれを改善", { x: MX, y: 1.05, w: CW, h: 0.35, fontSize: 15, color: SLATE });

	const stats = [
		["ソルバー", "Heunの2次法", "決定論的なODEで生成"],
		["ステップ数", "18 steps", "＝ 35 NFE"],
		["品質（CIFAR-10）", "FID 1.79", "class-conditional"],
	];
	const cw = (CW - 0.6) / 3;
	stats.forEach(([label, big, small], i) => {
		const x = MX + i * (cw + 0.3);
		card(s, x, 1.6, cw, 1.4, TINT, label);
		text(s, label, { x: x + 0.25, y: 1.72, w: cw - 0.5, h: 0.3, fontSize: 12, color: MUTED });
		text(s, big, { x: x + 0.25, y: 2.03, w: cw - 0.5, h: 0.55, fontSize: 26, bold: true, color: NAVY });
		text(s, small, { x: x + 0.25, y: 2.6, w: cw - 0.5, h: 0.3, fontSize: 13 });
	});

	const fw = 9.4;
	const fh = picture(s, "edm_fig2.png", MX, 3.25, fw, "figure EDM Fig 2");
	text(s, "決定論的サンプラーの比較（横軸：NFE、縦軸：FID）。EDMの改良を加えるほど（赤）、少ないNFEで低いFIDに届く。", { x: MX, y: 3.25 + fh + 0.06, w: fw, h: 0.3, fontSize: 11, color: MUTED });

	card(s, MX + fw + 0.25, 3.25, W - MX - (MX + fw + 0.25), fh, NAVY, "our base");
	text(s, [
		{ text: "本研究の土台", options: { bold: true, fontSize: 15, breakLine: true } },
		{ text: "学習・生成とも、EDMの設定（DDPM++、Heun 18 steps）を使用", options: { fontSize: 12.5, color: C.background2 } },
	], { x: MX + fw + 0.4, y: 3.42, w: W - MX - (MX + fw + 0.4) - 0.15, h: fh - 0.3, color: C.background1, paraSpaceAfter: 6 });
	source(s, "Karras et al., 2022（EDM）, Fig. 2");

	s.addNotes("2022年のEDMは、それまでばらばらだった拡散モデルの設計要素、つまりノイズの付け方やスケジュール、ネットワークの前処理、ソルバーを1つの枠組みで整理し、それぞれを改善した研究です。ソルバーにはHeunの2次法を使い、18ステップ、ネットワーク評価35回でCIFAR-10のFID 1.79を達成しました。下の図のように、改良を重ねるほど少ない計算で良いFIDに届きます。本研究の学習と生成は、このEDMの設定をそのまま土台にしています。");
}

// ---------------------------------------------------------------------------- 6. NFE and FID

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景" });
	s.addText("評価の物差し：NFEとFID", { placeholder: "title" });

	const cw = (CW - 0.4) / 2, y = 1.25, h = 4.55;
	// NFE
	card(s, MX, y, cw, h, TINT, "NFE");
	chip(s, "NFE（計算量）", MX + 0.25, y + 0.2, 1.7, BLUE);
	text(s, "1枚の画像を作るのに、ネットワークを呼ぶ回数", { x: MX + 0.25, y: y + 0.65, w: cw - 0.5, h: 0.35, fontSize: 15, bold: true, color: NAVY });
	text(s, "Heunの2次法は、1ステップでネットワークを2回呼ぶ（最後のステップだけ1回）", { x: MX + 0.25, y: y + 1.1, w: cw - 0.5, h: 0.6, fontSize: 13 });
	// Step boxes: 1, 2, ..., 18 with dots for evaluations.
	const bx = MX + 0.35, by = y + 1.95, bw = 0.62, gap = 0.12;
	const steps = [["1", 2], ["2", 2], ["3", 2], ["…", 0], ["17", 2], ["18", 1]];
	steps.forEach(([label, n], i) => {
		const x = bx + i * (bw + gap);
		if (n === 0) {
			text(s, "…", { x, y: by + 0.1, w: bw, h: 0.4, fontSize: 20, color: MUTED, align: "center" });
			return;
		}
		s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: 0.62, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: BLUE, width: 1 }, objectName: `step ${label}` });
		for (let d = 0; d < n; d++) {
			const dx = n === 2 ? x + 0.13 + d * 0.22 : x + 0.24;
			s.addShape(pres.shapes.OVAL, { x: dx, y: by + 0.12, w: 0.14, h: 0.14, fill: { color: BLUE }, line: { type: "none" }, objectName: `eval ${label}-${d}` });
		}
		text(s, label, { x, y: by + 0.33, w: bw, h: 0.25, fontSize: 10, color: MUTED, align: "center" });
	});
	text(s, "ステップ", { x: bx, y: by + 0.7, w: 3, h: 0.25, fontSize: 10.5, color: MUTED });
	text(s, [
		{ text: "18 steps → 2 × 18 − 1 = ", options: { color: C.text1 } },
		{ text: "35 NFE", options: { bold: true, color: BLUE } },
	], { x: MX + 0.25, y: y + 3.15, w: cw - 0.5, h: 0.45, fontSize: 20, align: "center" });
	text(s, "NFEが多いほど品質は上がりやすいが、遅くなる", { x: MX + 0.25, y: y + 3.8, w: cw - 0.5, h: 0.35, fontSize: 13, color: MUTED, align: "center" });

	// FID
	const fx = MX + cw + 0.4;
	card(s, fx, y, cw, h, TINT, "FID");
	chip(s, "FID（品質）", fx + 0.25, y + 0.2, 1.5, GREEN);
	text(s, "生成画像と実画像の、特徴の分布の距離", { x: fx + 0.25, y: y + 0.65, w: cw - 0.5, h: 0.35, fontSize: 15, bold: true, color: NAVY });
	text(s, "画像認識ネットワークで特徴を取り出し、2つの分布のずれを測る", { x: fx + 0.25, y: y + 1.1, w: cw - 0.5, h: 0.6, fontSize: 13 });
	const ex = fx + 1.0, ey = y + 1.75;
	s.addShape(pres.shapes.OVAL, { x: ex, y: ey, w: 2.1, h: 1.05, fill: { color: GREEN, transparency: 75 }, line: { color: GREEN, width: 1.5 }, objectName: "real distribution" });
	s.addShape(pres.shapes.OVAL, { x: ex + 1.35, y: ey + 0.15, w: 2.1, h: 1.05, fill: { color: ORANGE, transparency: 75 }, line: { color: ORANGE, width: 1.5 }, objectName: "generated distribution" });
	text(s, "実画像", { x: ex + 0.1, y: ey + 0.38, w: 0.9, h: 0.3, fontSize: 12, bold: true, color: GREEN });
	text(s, "生成画像", { x: ex + 2.4, y: ey + 0.53, w: 1.0, h: 0.3, fontSize: 12, bold: true, color: ORANGE });
	text(s, [
		{ text: "低いほど良い", options: { bold: true, color: GREEN } },
		{ text: "（0なら分布が一致）" },
	], { x: fx + 0.25, y: y + 3.15, w: cw - 0.5, h: 0.45, fontSize: 18, align: "center" });
	text(s, "50,000枚で計算（FID-50k）。枚数が少ないと値がばらつく", { x: fx + 0.25, y: y + 3.8, w: cw - 0.5, h: 0.35, fontSize: 13, color: MUTED, align: "center" });

	card(s, MX, 6.05, CW, 0.62, NAVY, "fair comparison");
	text(s, "手法を比べるときは、同じNFE（計算量）で、どれだけFIDが良いかを見る", { x: MX + 0.3, y: 6.13, w: CW - 0.6, h: 0.45, fontSize: 15, bold: true, color: C.background1, valign: "middle" });

	s.addNotes("ここで、評価に使う2つの物差しを確認します。NFEは、1枚の画像を作るのにネットワークを何回呼ぶかで、計算量の目安です。EDMで使うHeunの2次法は1ステップで2回呼ぶので、18ステップなら最後の1回を除いて35回、35 NFEになります。FIDは、生成画像と実画像の特徴の分布がどれだけずれているかで、低いほど良い指標です。本研究では5万枚で計算していて、枚数が少ないと値がばらつきます。NFEを増やせば品質は上がりやすいので、手法を比べるときは、同じNFEでどれだけFIDが良いかを見るのが大事です。");
}

// ---------------------------------------------------------------------------- 7. Restart Sampling

pres.addSection({ title: "Restart" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Restart" });
	s.addText("Restart Sampling（Xu et al., 2023）", { placeholder: "title" });
	text(s, "ODEの速さと、SDEの誤差を打ち消す働きの「いいとこ取り」", { x: MX, y: 1.05, w: CW, h: 0.35, fontSize: 16, color: SLATE });

	const fw = 8.1;
	const fh = picture(s, "restart_fig1.png", MX, 1.6, fw, "figure Restart Fig 1");
	text(s, "(a) 各手法の進め方（橙：ノイズ除去、緑：ノイズ追加）。(b) NFEと品質の関係（下ほど良い）：ODEは少ないNFE、SDEは多いNFEで強く、Restartはどちらでも良い。",
		{ x: MX, y: 1.6 + fh + 0.06, w: fw, h: 0.55, fontSize: 11, color: MUTED });

	const rx = MX + fw + 0.35, rw = W - MX - rx;
	const items = [
		["ODE", BLUE, "速いが、途中の誤差が溜まる"],
		["SDE", ORANGE, "誤差を打ち消せるが、遅い"],
	];
	items.forEach(([label, color, body], i) => {
		const y = 1.6 + i * 0.85;
		card(s, rx, y, rw, 0.7, TINT, label);
		chip(s, label, rx + 0.2, y + 0.2, 0.8, color);
		text(s, body, { x: rx + 1.15, y: y + 0.21, w: rw - 1.3, h: 0.3, fontSize: 13 });
	});
	card(s, rx, 3.35, rw, 3.25, NAVY, "restart");
	chip(s, "Restart", rx + 0.2, 3.52, 1.0, C.background1, NAVY);
	text(s, [
		{ text: "ODEで下りる途中、区間 [t_min, t_max] で", options: { breakLine: true } },
		{ text: "① ノイズを大きく加えて t_max まで戻り", options: { breakLine: true, bold: true } },
		{ text: "② ODEで t_min まで下り直す", options: { breakLine: true, bold: true } },
		{ text: "これを K 回繰り返す", options: { breakLine: true } },
		{ text: "モデルは同じで、生成の手順だけを変える。NFEはKに比例して少しずつ増える", options: { fontSize: 12, color: C.background2 } },
	], { x: rx + 0.2, y: 3.95, w: rw - 0.4, h: 2.55, fontSize: 13.5, color: C.background1, paraSpaceAfter: 5 });
	source(s, "Xu et al., 2023（Restart Sampling）, Fig. 1");

	s.addNotes("今回扱っているRestart Samplingは、ODEとSDEのいいとこ取りを狙った手法です。ODEは速いですが途中の誤差が溜まり、SDEは誤差を打ち消せますが遅い、という性質がありました。Restartは、ODEで下りる途中のある区間で、ノイズを大きく加えて一度上に戻り、そこからまたODEで下り直します。左の図(a)の緑の矢印が戻る部分で、これをK回繰り返します。ノイズを加えて戻ることで、SDEのように溜まった誤差を打ち消せます。モデルは同じで、生成の手順だけを変える方法です。図(b)のように、少ない計算量でも多い計算量でも良い品質が出るとされています。");
}

// ---------------------------------------------------------------------------- 8. Restart parameters
// Values follow generate.py (restart_sampler / restart_nfe) and the experiment READMEs.

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Restart" });
	s.addText("Restartのパラメーター（本研究の設定）", { placeholder: "title" });

	// Left: one Restart loop, magnified. y grows downward = less noise.
	card(s, MX, 1.2, 6.75, 4.75, TINT, "one restart loop");
	const ax = 1.35, axisBottom = 5.55;
	seg(s, ax, axisBottom, ax, 1.35, MUTED, 1, { endArrowType: "triangle" });
	seg(s, ax, axisBottom, 7.05, axisBottom, MUTED, 1, { endArrowType: "triangle" });
	text(s, "ノイズ量 σ", { x: MX + 0.05, y: 1.3, w: 0.68, h: 0.5, fontSize: 10.5, color: MUTED, align: "right" });
	text(s, "生成の進行", { x: 6.05, y: 5.22, w: 1.0, h: 0.25, fontSize: 10.5, color: MUTED, align: "right" });

	const yMax = 2.9, yMin = 4.1;
	seg(s, ax, yMax, 6.95, yMax, MUTED, 1, { dashType: "dash" });
	seg(s, ax, yMin, 6.95, yMin, MUTED, 1, { dashType: "dash" });
	text(s, "t_max = 0.30", { x: 5.6, y: yMax - 0.3, w: 1.35, h: 0.25, fontSize: 12, bold: true, color: NAVY, align: "right" });
	text(s, "t_min = 0.14", { x: 5.6, y: yMin - 0.3, w: 1.35, h: 0.25, fontSize: 12, bold: true, color: NAVY, align: "right" });

	const dot = (x, y, color, name) => s.addShape(pres.shapes.OVAL, { x: x - 0.07, y: y - 0.07, w: 0.14, h: 0.14, fill: { color }, line: { type: "none" }, objectName: name });
	const path = (pts, color, name) => {
		pts.slice(1).forEach((p, i) => seg(s, pts[i][0], pts[i][1], p[0], p[1], color, 2.5));
		pts.forEach((p, i) => dot(p[0], p[1], color, `${name} ${i}`));
	};
	const snap = [3.6, 4.25]; // the main-schedule point closest to t_min
	path([[1.6, 1.75], [2.1, 2.4], [2.6, 3.05], [3.1, 3.65], snap], BLUE, "main path");
	seg(s, snap[0], snap[1] - 0.08, snap[0], yMax + 0.08, GREEN, 2.5, { endArrowType: "triangle" });
	path([[snap[0], yMax], [4.15, 3.6], [4.7, 4.25]], ORANGE, "restart steps");
	path([[4.7, 4.25], [5.3, 4.8], [5.9, 5.3]], BLUE, "main path to zero");

	text(s, [
		{ text: "ノイズを加えて t_max へ戻る", options: { bold: true, color: GREEN, breakLine: true } },
		{ text: "標準偏差 √(t_max² − t_min²)", options: { fontSize: 10.5, color: MUTED } },
	], { x: 3.75, y: 2.15, w: 2.6, h: 0.6, fontSize: 12 });
	text(s, [
		{ text: "N_restart = 3 段", options: { bold: true, color: ORANGE, breakLine: true } },
		{ text: "Heun 2ステップ（4 NFE）", options: { fontSize: 10.5, color: MUTED } },
	], { x: 4.75, y: 3.15, w: 2.5, h: 0.6, fontSize: 12 });
	text(s, "主経路上で t_min に最も近い点（actual t_min）", { x: 1.45, y: 4.35, w: 1.95, h: 0.5, fontSize: 10.5, color: MUTED, align: "right" });
	seg(s, snap[0], 4.55, 4.7, 4.55, NAVY, 1.25, { beginArrowType: "triangle", endArrowType: "triangle" });
	text(s, "この往復を K 回", { x: 3.5, y: 4.62, w: 1.3, h: 0.25, fontSize: 10.5, bold: true, color: NAVY, align: "center" });
	[["主経路（ODE）", BLUE], ["ノイズ追加", GREEN], ["下り直し（ODE）", ORANGE]].forEach(([label, color], i) => {
		const lx = MX + 0.3 + i * 2.05;
		seg(s, lx, 5.78, lx + 0.4, 5.78, color, 2.5);
		text(s, label, { x: lx + 0.48, y: 5.66, w: 1.5, h: 0.25, fontSize: 10.5 });
	});

	// Right: parameter table.
	const tx = MX + 6.75 + 0.3, tw = W - MX - tx;
	const head = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: NAVY }, align: "center", fontSize: 12 } });
	const sym = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 11.5 } });
	const rows = [
		[head("記号"), head("意味"), head("本研究")],
		[sym("K"), "戻って下り直す回数", "0〜8"],
		[sym("t_min"), "戻り始めるノイズ量（区間の下端）", "0.14（比較で0.06）"],
		[sym("t_max"), "戻る先のノイズ量（区間の上端）", "0.30"],
		[sym("N_restart"), "下り直すときの段数", "3（2ステップ）"],
		[sym("N_main"), "主経路のステップ数", "18（計算量固定では 28−2K）"],
		[sym("restart_noise"), "加えるノイズの倍率", "1"],
	];
	s.addTable(rows, {
		x: tx, y: 1.2, w: tw, colW: [1.55, 2.1, tw - 3.65], rowH: [0.4, 0.5, 0.62, 0.62, 0.5, 0.62, 0.5],
		fontSize: 12, color: C.text1, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});
	text(s, "参考：論文著者が公開しているEDM・CIFAR-10向けの設定は N_restart = 3、K = 2、t_min = 0.14、t_max = 0.30", { x: tx, y: 5.1, w: tw, h: 0.6, fontSize: 11, color: MUTED });

	card(s, MX, 6.08, CW, 0.7, NAVY, "nfe formula");
	text(s, [
		{ text: "NFE = (2 × N_main − 1) + K × 2 × (N_restart − 1)", options: { bold: true } },
		{ text: "　→　(2×18 − 1) + K × 2 × (3 − 1) = ", options: { color: C.background2 } },
		{ text: "35 + 4K", options: { bold: true } },
	], { x: MX + 0.3, y: 6.15, w: CW - 0.6, h: 0.55, fontSize: 16, color: C.background1, valign: "middle" });
	source(s, "Xu et al., 2023（Restart Sampling）、本研究の実装（generate.py）");

	s.addNotes("実験で動かしているRestartのパラメーターを確認します。左の図はRestart 1回分を拡大したものです。主経路のODEで下りてきて、t_min付近まで来たらノイズを加え、t_maxまで戻ります。そこからN_restart = 3段、つまりHeunの2ステップでt_minまで下り直します。この往復をK回繰り返してから、主経路で最後まで下ります。本研究ではt_maxを0.30、t_minを0.14に固定し、Kを0から8まで変えています。1回の往復でネットワークを4回呼ぶので、NFEは主経路の35に4Kを足した35+4Kになります。注意点として、t_minは主経路のノイズ量の中で最も近い点に丸められます。主経路のステップ数を変えると実際の開始点もずれるので、進捗報告ではこのactual t_minも示します。");
}

// ---------------------------------------------------------------------------- 9. research question

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Restart" });
	s.addText("本研究の問い", { placeholder: "title" });

	const fw = 4.1;
	const fh = picture(s, "restart_fig5.png", MX, 1.3, fw, "figure Restart Fig 5");
	text(s, "Kを増やすとFIDはいったん改善し、その後悪化する。強いモデル（EDM）ほど、悪化に転じるKが小さい。", { x: MX, y: 1.3 + fh + 0.08, w: fw, h: 0.8, fontSize: 11.5, color: MUTED });

	const rx = MX + fw + 0.45, rw = W - MX - rx;
	card(s, rx, 1.3, rw, 1.25, TINT, "paper finding");
	chip(s, "論文の知見", rx + 0.25, 1.45, 1.3, SLATE);
	text(s, "Restartの効果と最適なKは、モデルの強さ（誤差の大きさ）によって変わる", { x: rx + 0.25, y: 1.88, w: rw - 0.5, h: 0.55, fontSize: 15, bold: true, color: NAVY });

	chip(s, "本研究の問い", rx, 2.85, 1.45);
	const questions = [
		"学習途中のモデル（誤差が大きい）ほど、Restartは効くか？",
		"最適なKは、学習段階によってどう変わるか？",
		"計算量（NFE）をそろえても、Restartは有効か？",
	];
	questions.forEach((q, i) => {
		const y = 3.35 + i * 0.72;
		s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: rx, y, w: 0.45, h: 0.45, fill: { color: NAVY }, line: { type: "none" }, color: C.background1, fontSize: 15, bold: true, align: "center", valign: "middle", margin: 0, objectName: `question ${i + 1}` });
		text(s, q, { x: rx + 0.65, y: y + 0.07, w: rw - 0.65, h: 0.4, fontSize: 16 });
	});

	card(s, rx, 5.7, rw, 0.95, NAVY, "approach");
	text(s, [
		{ text: "→ CIFAR-10でEDMを再学習し、10kごとのcheckpointで検証", options: { bold: true, breakLine: true } },
		{ text: "ここから進捗報告", options: { fontSize: 12.5, color: C.background2 } },
	], { x: rx + 0.3, y: 5.8, w: rw - 0.6, h: 0.75, fontSize: 15, color: C.background1, valign: "middle", paraSpaceAfter: 3 });
	source(s, "Xu et al., 2023（Restart Sampling）, Fig. 5");

	s.addNotes("Restartの論文では、繰り返し回数Kを増やすとFIDはいったん良くなり、その後悪くなることが示されています。さらに、強いモデルであるEDMの方が、悪化に転じるKが小さくなっています。つまり、Restartの効果や最適なKは、モデルの誤差の大きさによって変わると考えられます。そこで本研究では、学習途中で誤差が大きいモデルほどRestartが効くのか、最適なKは学習段階によってどう変わるのか、そして計算量をそろえてもRestartは有効なのか、を調べています。CIFAR-10でEDMを再学習し、10kごとのcheckpointで検証しました。ここから進捗報告に入ります。");
}

// ---------------------------------------------------------------------------- write

// Theme fonts from pptxgenjs cover Latin text only; point Japanese text at the same font.
async function setEastAsianFont(file) {
	const zip = await JSZip.loadAsync(fs.readFileSync(file));
	for (const name of Object.keys(zip.files).filter((f) => /^ppt\/theme\/theme\d+\.xml$/.test(f))) {
		let xml = await zip.file(name).async("string");
		xml = xml.replace(/<a:ea typeface="[^"]*"\/>/g, `<a:ea typeface="${FONT}"/>`)
			.replace(/<a:font script="Jpan" typeface="[^"]*"\/>/g, `<a:font script="Jpan" typeface="${FONT}"/>`);
		zip.file(name, xml);
	}
	fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
	await pres.writeFile({ fileName: OUT });
	await applyTheme(OUT, THEME);
	await setEastAsianFont(OUT);
	console.log(`wrote ${OUT}`);
})();
