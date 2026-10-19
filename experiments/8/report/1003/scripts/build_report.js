// Build the Restart Sampling progress report deck.
//
// Usage (from the EDM repository root, after scripts/crop_figures.py):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/1003/scripts/build_report.js
//
// Values on the slides come from the result TSVs of the Restart experiments and the
// retraining checkpoint directory; update the PROGRESS block when those change.

const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report", "1003");
const ASSETS = path.join(REPORT_DIR, "assets");
const OUT = path.join(REPORT_DIR, "restart_sampling_progress_report.pptx");

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

// Latest saved range of the retraining run, checked when the deck is built.
const PROGRESS = {
	savedLast: "160k",
	savedCount: 16,
	totalCount: 20,
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "研究進捗";
const C = pres.SchemeColor;
const ORANGE = C.accent2, BLUE = C.accent3, GREEN = C.accent4, NAVY = C.text2, TINT = C.background2;

const W = 13.333, MX = 0.55, CW = W - 2 * MX;

// ---------------------------------------------------------------------------- layouts

pres.defineSlideMaster({
	title: "TITLE",
	background: { color: THEME.colors.dk2 },
	objects: [
		{ placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.2, w: 11.5, h: 1.3, fontSize: 48, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
		{ placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.75, w: 11.5, h: 0.8, fontSize: 22, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
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

// Straight segment from (x1, y1) to (x2, y2); the arrow head, if any, sits at (x2, y2).
function seg(slide, x1, y1, x2, y2, color, width, extra) {
	const opts = Object.assign({
		x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
		line: { color, width },
	}, extra && extra.shape);
	if (extra && extra.line) Object.assign(opts.line, extra.line);
	if ((x2 < x1) !== (y2 < y1)) opts.flipV = true;
	if (x2 < x1) opts.flipH = !opts.flipH;
	slide.addShape(pres.shapes.LINE, opts);
}

function pictureFit(slide, file, x, y, w, aspect, name) {
	slide.addImage({ path: path.join(ASSETS, file), x, y, w, h: w / aspect, objectName: name });
	return w / aspect;
}

// ---------------------------------------------------------------------------- 1. title

pres.addSection({ title: "導入" });
{
	const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "導入" });
	s.addText("研究進捗", { placeholder: "title" });
	s.addText("CIFAR-10 class-conditional EDMにおける再学習、Restart実験", { placeholder: "body" });

	// Faint Restart trajectory as the cover motif: denoise, then noise-up / re-denoise loops.
	const top = 5.15, tmax = 5.95, tmin = 6.35, bottom = 6.75;
	const faint = { line: { transparency: 55 } };
	seg(s, 7.0, top, 9.6, tmin, C.background2, 2.25, faint);
	for (let i = 0; i < 3; i++) {
		const x = 9.6 + i * 0.6;
		seg(s, x, tmin, x, tmax, ORANGE, 2.25, { line: { transparency: 25 } });
		seg(s, x, tmax, x + 0.6, tmin, GREEN, 2.25, { line: { transparency: 25 } });
	}
	seg(s, 11.4, tmin, 12.4, bottom, C.background2, 2.25, faint);
	s.addNotes("CIFAR-10 class-conditional EDMについて、10k間隔のcheckpoint系列を作るための再学習と、Restart Samplingの評価の進捗を報告します。");
}

// ---------------------------------------------------------------------------- 2. Restart basics

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "導入" });
	s.addText("Restart Samplingの仕組みと比較指標", { placeholder: "title" });

	// Schematic: normal ODE (upper) and Restart (lower) on the same axes.
	const x0 = 2.35, xEndOde = 5.4;
	function panel(py, label, restart) {
		card(s, MX, py, 7.25, 1.95, TINT, `schematic ${label}`);
		chip(s, label, MX + 0.2, py + 0.15, 1.2);
		const top = py + 0.3, tmax = py + 0.85, tmin = py + 1.2, bottom = py + 1.5, axisY = py + 1.62;
		const slope = (bottom - top) / (xEndOde - x0);
		const xTmin = x0 + (tmin - top) / slope;
		if (restart) {
			s.addShape(pres.shapes.RECTANGLE, { x: x0, y: tmax, w: 5.25, h: tmin - tmax, fill: { color: ORANGE, transparency: 88 }, line: { type: "none" }, objectName: "restart interval" });
			text(s, "Restart区間", { x: 6.45, y: tmax + 0.06, w: 1.15, h: 0.25, fontSize: 10, color: ORANGE, bold: true, align: "right" });
		}
		seg(s, x0 - 0.15, axisY, 7.6, axisY, MUTED, 1, { line: { endArrowType: "triangle" } });
		seg(s, x0 - 0.15, axisY, x0 - 0.15, top - 0.12, MUTED, 1, { line: { endArrowType: "triangle" } });
		text(s, "ノイズ大", { x: MX + 0.2, y: top + 0.25, w: 1.35, h: 0.25, fontSize: 10.5, color: MUTED, align: "right" });
		text(s, "ノイズ小", { x: MX + 0.2, y: bottom - 0.2, w: 1.35, h: 0.25, fontSize: 10.5, color: MUTED, align: "right" });
		seg(s, x0, top, xTmin, tmin, BLUE, 3);
		if (!restart) {
			seg(s, xTmin, tmin, xEndOde, bottom, BLUE, 3);
			text(s, "35 NFE", { x: xEndOde + 0.1, y: bottom - 0.3, w: 1.0, h: 0.25, fontSize: 11, bold: true, color: BLUE });
			return;
		}
		const loopW = 0.45;
		for (let i = 0; i < 3; i++) {
			const x = xTmin + i * loopW;
			seg(s, x, tmin, x, tmax, ORANGE, 3, { line: { endArrowType: "triangle" } });
			seg(s, x, tmax, x + loopW, tmin, GREEN, 3);
		}
		const xLoopEnd = xTmin + 3 * loopW;
		seg(s, xLoopEnd, tmin, xLoopEnd + (bottom - tmin) / slope, bottom, BLUE, 3);
		text(s, "K回繰り返す", { x: xTmin - 0.2, y: tmax - 0.33, w: 3 * loopW + 0.4, h: 0.25, fontSize: 10.5, bold: true, color: C.text1, align: "center" });
		text(s, "35+4K NFE", { x: xLoopEnd + (bottom - tmin) / slope + 0.08, y: bottom - 0.3, w: 0.95, h: 0.25, fontSize: 10.5, bold: true, color: ORANGE });
		text(s, "生成の進行（モデル呼び出し）", { x: 4.3, y: axisY + 0.03, w: 3.3, h: 0.25, fontSize: 10, color: MUTED, align: "right" });
	}
	panel(1.2, "通常ODE", false);
	panel(3.3, "Restart", true);

	// Legend for the three segment colors.
	const ly = 5.43;
	[["ノイズ除去", BLUE, 0.55], ["ノイズ追加", ORANGE, 2.55], ["再度のノイズ除去", GREEN, 4.55]].forEach(([label, color, x]) => {
		seg(s, x + 0.2, ly + 0.12, x + 0.65, ly + 0.12, color, 3);
		text(s, label, { x: x + 0.75, y: ly, w: 1.9, h: 0.25, fontSize: 11 });
	});

	// Terms.
	const terms = [
		["通常ODE", "ノイズを段階的に減らして画像を生成する。"],
		["Restart", "生成途中でノイズを追加し、その区間をもう一度ノイズ除去する。使用するモデルは同じで、生成の進め方を変える。"],
		["K", "「ノイズ追加 → ノイズ除去」を繰り返す回数。K=0はRestartなし。"],
		["NFE", "画像1枚の生成でモデルを呼び出す回数。計算量の目安。"],
		["FID", "生成画像の品質を評価する指標。低いほど良い。"],
	];
	const tx = 8.1, ty = [1.25, 1.95, 2.95, 3.75, 4.5];
	terms.forEach(([term, desc], i) => {
		text(s, term, { x: tx, y: ty[i], w: 1.1, h: 0.3, fontSize: 14, bold: true, color: NAVY });
		text(s, desc, { x: tx + 1.15, y: ty[i] + 0.01, w: 3.53, h: 0.75, fontSize: 12.5 });
	});

	card(s, MX, 5.85, CW, 0.9, TINT, "setting");
	text(s, [
		{ text: "今回の設定：", options: { bold: true, color: NAVY } },
		{ text: "通常ODEは35 NFE、Restartは35+4K NFE。K=5では55 NFE", options: { bold: true, breakLine: true } },
		{ text: "→ Restartは計算量も増えるため、同じ55 NFEの通常ODE（28 steps）とも比較し、改善が計算量の増加だけによるものでないかを確認する", options: { fontSize: 12.5 } },
	], { x: MX + 0.25, y: 5.97, w: CW - 0.5, h: 0.7, fontSize: 14, paraSpaceAfter: 4 });

	s.addNotes([
		"通常のODEサンプラーはノイズを段階的に減らして画像を生成します。Restartは、生成途中のRestart区間でノイズを加えて少し戻し、その区間をもう一度ノイズ除去します。モデルは同じで、生成の進め方だけを変えます。",
		"Kはこの「ノイズ追加→ノイズ除去」を繰り返す回数で、K=0はRestartなしです。NFEは1枚あたりのモデル呼び出し回数で計算量の目安、FIDは低いほど良い品質指標です。",
		"今回の設定では、通常ODE（18 steps）が35 NFE、Restartは35+4K NFEで、K=5なら55 NFEです。Restartは計算量も増えるので、同じ55 NFEになるよう28 stepsにした通常ODEとも比較し、改善が計算量の増加だけで説明できないかを確認します。",
		"Restartの設定：Restart区間 t_min=0.06, t_max=0.30、区間内3 steps、noise multiplier 1。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 3. existing model

pres.addSection({ title: "実験結果" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText("既存モデルでのRestart効果の検証", { placeholder: "title" });

	const y = 1.2, h = 1.48;
	card(s, MX, y, 4.0, h, TINT, "purpose and setup");
	chip(s, "目的・条件", MX + 0.2, y + 0.14, 1.25);
	text(s, [
		{ text: "5.5k・20k・120k・200k（初期と後期）", options: { breakLine: true } },
		{ text: "K=5、各条件50,000枚", options: { breakLine: true } },
		{ text: "独立した生成seed範囲で3回評価" },
	], { x: MX + 0.2, y: y + 0.55, w: 3.65, h: 0.85, fontSize: 12.5, paraSpaceAfter: 3 });

	card(s, 4.75, y, 3.3, h, TINT, "comparison");
	chip(s, "比較", 4.95, y + 0.14, 0.8);
	text(s, [
		{ text: "通常ODE（35 NFE）", options: { breakLine: true } },
		{ text: "Restart（55 NFE）", options: { breakLine: true, bold: true, color: ORANGE } },
		{ text: "同一計算量ODE（55 NFE）" },
	], { x: 4.95, y: y + 0.55, w: 3.0, h: 0.85, fontSize: 12.5, paraSpaceAfter: 3 });

	const rx = 8.25, rw = W - MX - rx;
	card(s, rx, y, rw, h, TINT, "result");
	chip(s, "結果", rx + 0.2, y + 0.14, 0.8);
	text(s, "同一NFEのODEに対する改善幅（3回平均）", { x: rx + 1.1, y: y + 0.16, w: rw - 1.2, h: 0.28, fontSize: 11, color: MUTED });
	[["5.5k", "+1.718", GREEN], ["20k", "+0.318", GREEN], ["120k", "−0.048", C.accent6], ["200k", "−0.181", C.accent6]].forEach(([ckpt, value, color], i) => {
		const x = rx + 0.2 + i * 1.08;
		text(s, value, { x, y: y + 0.52, w: 1.05, h: 0.4, fontSize: 16, bold: true, color, align: "center" });
		text(s, ckpt, { x, y: y + 0.92, w: 1.05, h: 0.22, fontSize: 10.5, color: MUTED, align: "center" });
	});
	text(s, "初期では改善し、後期では悪化した", { x: rx + 0.2, y: y + 1.15, w: rw - 0.4, h: 0.28, fontSize: 12, bold: true, color: NAVY, align: "center" });

	text(s, "図：既存モデルにおけるRestart K=5のFID-50,000（3回の独立評価）", { x: MX, y: 2.88, w: CW, h: 0.3, fontSize: 13, bold: true, color: NAVY });
	const fw = 10.4;
	const fh = pictureFit(s, "existing_model_fid50k.png", (W - fw) / 2, 3.22, fw, 2380 / 741, "figure existing model FID-50k");
	text(s, "各条件50,000枚 × 3回（seed 0–49,999／50,000–99,999／100,000–149,999）、エラーバーは標本標準偏差。(a) 3回平均のFID（低いほど良い）。(b) 改善幅 = ODEのFID − RestartのFID（正値が改善）。黄：通常ODE（35 NFE）との差、緑：同一NFE（55 NFE）のODEとの差。",
		{ x: MX, y: 3.22 + fh + 0.08, w: CW, h: 0.5, fontSize: 10.5, color: MUTED });

	s.addNotes([
		"まず既存の学習済みモデルで、学習初期と後期のcheckpointにRestartを適用しました。checkpointは5.5k、20k、120k、200kで、K=5、各条件50,000枚のFIDを、重ならない3つの生成seed範囲で3回評価しています。",
		"比較対象は、通常ODE（35 NFE）、Restart（55 NFE）、そして計算量をそろえた通常ODE（28 steps、55 NFE）です。",
		"同一NFEのODEに対する改善幅は、5.5kで+1.718、20kで+0.318、120kで−0.048、200kで−0.181でした。学習初期では改善しましたが、後期ではむしろ悪化しています。3回評価の標準偏差は0.01〜0.12程度で、符号はどのseed範囲でも同じでした。",
		"改善幅はすべて「ODEのFID − RestartのFID」で、正値がRestartの改善です。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 4. retraining

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText("学習段階を細かく比較するための再学習", { placeholder: "title" });

	chip(s, "目的", MX, 1.3, 0.8);
	text(s, [
		{ text: "これまで評価したcheckpoint（5.5k・20k・120k・200k）は間隔が不均一だった。", options: { breakLine: true } },
		{ text: "→ 同じ学習系列から、厳密な10k間隔のcheckpoint系列を作り直す。", options: { bold: true, color: NAVY } },
	], { x: MX, y: 1.75, w: 7.3, h: 0.95, fontSize: 15, paraSpaceAfter: 6 });

	chip(s, "条件", MX, 3.05, 0.8);
	text(s, [
		{ text: "CIFAR-10 class-conditional", options: { bullet: true, breakLine: true } },
		{ text: "DDPM++（ネットワーク）、EDM（前処理・損失）", options: { bullet: true, breakLine: true } },
		{ text: "学習率 0.001、EMA 0.5 Mimg", options: { bullet: true, breakLine: true } },
		{ text: "学習seed = 0", options: { bullet: true, breakLine: true } },
		{ text: "200kまで学習し、10kごとに保存（10k〜200kの20点）", options: { bullet: true } },
	], { x: MX, y: 3.5, w: 7.3, h: 2.8, fontSize: 16, paraSpaceAfter: 12 });

	const px = 8.35, pw = W - MX - px, py = 1.3, ph = 5.0;
	card(s, px, py, pw, ph, NAVY, "progress");
	chip(s, "進捗", px + 0.35, py + 0.35, 0.8, C.background1, NAVY);
	text(s, [
		{ text: PROGRESS.savedLast, options: { fontSize: 54, bold: true } },
		{ text: " / 200k", options: { fontSize: 22 } },
	], { x: px + 0.35, y: py + 0.95, w: pw - 0.7, h: 1.2, color: C.background1, valign: "bottom" });
	text(s, `まで保存済み（${PROGRESS.savedCount} / ${PROGRESS.totalCount}点）`, { x: px + 0.35, y: py + 2.25, w: pw - 0.7, h: 0.4, fontSize: 17, bold: true, color: C.background1 });
	text(s, "170k以降に向けて学習を継続中", { x: px + 0.35, y: py + 2.85, w: pw - 0.7, h: 0.35, fontSize: 14, color: C.background2 });
	text(s, "0から200kまで1プロセスで連続学習し、途中での再開による乱数列・データ順のリセットを避けている", { x: px + 0.35, y: py + 3.7, w: pw - 0.7, h: 0.8, fontSize: 12, color: C.background2 });

	s.addNotes([
		"これまで評価したcheckpointは5.5k、20k、120k、200kと間隔が不均一でした。学習段階による変化を細かく見るため、同じ条件で再学習し、厳密に10k間隔のcheckpoint系列を作っています。",
		"条件は既存の本番学習と同じで、CIFAR-10 class-conditional、DDPM++、EDM、学習率0.001、EMA 0.5 Mimg、学習seed 0です。200kまで学習し、10kごとに保存します。",
		`進捗：10k〜${PROGRESS.savedLast}のnetwork snapshotとtraining stateを保存済み（${PROGRESS.savedCount}/${PROGRESS.totalCount}点）。170k以降は学習継続中です。`,
		"保存タイミングの補足：batch=512のため、各保存点（例：10,000 kimg）は512で割り切れません。指定点に到達した最初のoptimizer更新の直後に保存しており、超過は最大384枚（0.384 kimg）です。ファイル名と評価上のラベルは10k刻みのままです。通常のtick単位の保存は無効化し、途中で再開しないことで乱数列とデータ順のリセットを避けています。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 5-6. K sweep

const K_SWEEP = {
	"10k": [7, 63, "+0.797"], "20k": [6, 59, "+0.378"], "30k": [3, 47, "+0.280"], "40k": [4, 51, "+0.165"], "50k": [5, 55, "+0.109"],
	"60k": [5, 55, "+0.168"], "70k": [1, 39, "+0.054"], "80k": [7, 63, "+0.130"], "90k": [7, 63, "+0.085"], "100k": [2, 43, "+0.062"],
};
const PANEL_ASPECT = { "10k": 795 / 504, "20k": 791 / 504, "30k": 791 / 504, "40k": 790 / 504, "50k": 795 / 512, "60k": 791 / 512, "70k": 791 / 512, "80k": 790 / 512, "90k": 795 / 522, "100k": 791 / 522 };

function kSweepSlide(page, ckpts, header, notes) {
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText(`学習段階ごとの最適なRestart回数の探索（${page}）`, { placeholder: "title" });

	header.forEach(([label, body], i) => {
		chip(s, label, MX, 1.13 + i * 0.4, 0.8);
		text(s, body, { x: MX + 0.95, y: 1.15 + i * 0.4, w: 8.2, h: 0.3, fontSize: 13 });
	});
	// Legend for the panel marks.
	const lx = 9.85;
	seg(s, lx, 1.28, lx + 0.45, 1.28, "6C757D", 1.25, { line: { dashType: "dash" } });
	text(s, "Restartなし（K=0）", { x: lx + 0.55, y: 1.15, w: 2.4, h: 0.28, fontSize: 12 });
	s.addShape(pres.shapes.STAR_5_POINT, { x: lx + 0.1, y: 1.5, w: 0.25, h: 0.25, fill: { color: ORANGE }, line: { type: "none" }, objectName: "legend star" });
	text(s, "最小FID（最良K）", { x: lx + 0.55, y: 1.55, w: 2.4, h: 0.28, fontSize: 12 });

	const pw = 3.55, gap = 0.3, x0 = (W - (3 * pw + 2 * gap)) / 2, rowY = [2.0, 4.42];
	ckpts.forEach((ckpt, i) => {
		const x = x0 + (i % 3) * (pw + gap), y = rowY[Math.floor(i / 3)];
		pictureFit(s, `k_sweep_${ckpt}.png`, x, y, pw, PANEL_ASPECT[ckpt], `FID vs K ${ckpt}`);
	});

	const cx = x0 + 2 * (pw + gap), cy = rowY[1], ch = 2.3;
	card(s, cx, cy, pw, ch, TINT, "best K summary");
	text(s, "このページの最良K（FID-5k）", { x: cx + 0.2, y: cy + 0.12, w: pw - 0.4, h: 0.28, fontSize: 12, bold: true, color: NAVY });
	const rows = [[
		{ text: "checkpoint", options: { bold: true, color: MUTED } },
		{ text: "最良K（NFE）", options: { bold: true, color: MUTED } },
		{ text: "改善幅", options: { bold: true, color: MUTED, align: "right" } },
	]].concat(ckpts.map((ckpt) => {
		const [k, nfe, gain] = K_SWEEP[ckpt];
		return [{ text: ckpt }, { text: `K=${k}（${nfe}）` }, { text: gain, options: { align: "right", color: GREEN, bold: true } }];
	}));
	s.addTable(rows, { x: cx + 0.2, y: cy + 0.45, w: pw - 0.4, colW: [1.0, 1.35, pw - 0.4 - 2.35], rowH: 0.22, fontSize: 10.5, color: C.text1, margin: [0.01, 0.03, 0.01, 0.03], border: { type: "none" } });
	text(s, "改善幅 = FID(K=0) − 最小FID。NFE = 35+4K。縦軸の範囲はパネルごとに異なる。", { x: cx + 0.2, y: cy + 1.83, w: pw - 0.4, h: 0.42, fontSize: 9.5, color: MUTED });

	s.addNotes(notes.join("\n\n"));
}

kSweepSlide("10k〜50k", ["10k", "20k", "30k", "40k", "50k"], [
	["目的", "最良のRestart回数Kが、学習段階によって変わるかを確認する"],
	["条件", "10k〜100kの再学習checkpoint、K=0〜8、各条件5,000枚、NFE = 35+4K（35〜67）"],
], [
	"再学習系列のcheckpointを使い、学習段階ごとにRestart回数Kを0から8まで振りました。各条件5,000枚のFIDで、NFEは35+4K、つまり35から67です。",
	"各パネルが1つのcheckpointで、横軸がK、破線がRestartなし（K=0）、星印がFID最小のKです。縦軸の範囲はパネルごとに異なるので、改善量は右下の表の値で比較してください。",
	"10kでは最良K=7で+0.797、20kではK=6で+0.378と、初期ほどRestartによる改善が大きくなっています。",
]);

kSweepSlide("60k〜100k", ["60k", "70k", "80k", "90k", "100k"], [
	["結果", "最良Kは一定ではなく、改善幅は学習初期ほど大きい"],
	["注意", "各条件5,000枚の評価のため、ここでの最良Kは候補選択として扱う"],
], [
	"60k〜100kでは、最良KがK=5、1、7、7、2と大きく変動し、K=0に対する改善幅も0.05〜0.17程度に小さくなりました。",
	"最小FIDと2番目のFIDの差は多くのcheckpointで0.01〜0.02程度しかなく、5,000枚のFIDの評価誤差を考えると、この結果だけで最良Kを確定することはできません。候補を絞るための結果として扱い、25,000枚の評価で確認します。",
]);

// ---------------------------------------------------------------------------- 7. uniform checkpoints

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText("均一なチェックポイントでRestart効果の再確認", { placeholder: "title" });

	const y = 1.2, h = 1.48, lw = 7.75;
	card(s, MX, y, lw, h, TINT, "purpose and setup");
	chip(s, "目的", MX + 0.2, y + 0.18, 0.8);
	text(s, "10k間隔で作り直した系列でも、学習に伴ってRestartの効果が小さくなるか確認する", { x: MX + 1.2, y: y + 0.2, w: lw - 1.4, h: 0.55, fontSize: 13 });
	chip(s, "条件", MX + 0.2, y + 0.82, 0.8);
	text(s, "K=5。通常ODE（35 NFE）、Restart（55 NFE）、同一計算量ODE（55 NFE）を比較。各条件25,000枚", { x: MX + 1.2, y: y + 0.84, w: lw - 1.4, h: 0.55, fontSize: 13 });

	const rx = MX + lw + 0.25, rw = W - MX - rx;
	card(s, rx, y, rw, h, NAVY, "result");
	text(s, "同一NFEのODEに対する改善幅", { x: rx + 0.25, y: y + 0.14, w: rw - 0.5, h: 0.28, fontSize: 12, color: C.background2 });
	const colW = (rw - 0.5 - 0.6) / 2;
	[["0.712", "10k", rx + 0.25], ["0.012", "100k", rx + 0.25 + colW + 0.6]].forEach(([value, ckpt, x]) => {
		text(s, value, { x, y: y + 0.45, w: colW, h: 0.6, fontSize: 30, bold: true, color: C.background1, align: "center", valign: "middle" });
		text(s, ckpt, { x, y: y + 1.07, w: colW, h: 0.25, fontSize: 11, color: C.background2, align: "center" });
	});
	text(s, "→", { x: rx + 0.25 + colW, y: y + 0.45, w: 0.6, h: 0.6, fontSize: 22, color: C.background1, align: "center", valign: "middle" });

	text(s, "図：再学習系列（10k間隔）におけるRestart K=5のFID-25,000（10k〜100k）", { x: MX, y: 2.88, w: CW, h: 0.3, fontSize: 13, bold: true, color: NAVY });
	const fw = 11.6;
	const fh = pictureFit(s, "uniform_checkpoints_fid25k.png", (W - fw) / 2, 3.22, fw, 3180 / 871, "figure uniform checkpoints FID-25k");
	text(s, "各条件25,000枚（seed 0–24,999、1回）。(a) FID（低いほど良い）。(b) 改善幅 = ODEのFID − RestartのFID（正値が改善）。黄：通常ODE（35 NFE）との差、緑：同一NFE（55 NFE）のODEとの差。110k以降はFID-25kの比較がそろっていないため未掲載。",
		{ x: MX, y: 3.22 + fh + 0.08, w: CW, h: 0.5, fontSize: 10.5, color: MUTED });

	s.addNotes([
		"再学習した10k間隔の系列で、既存モデルと同じK=5の比較を行いました。各条件25,000枚で、FID-25kの3条件がそろった10k〜100kを掲載しています。",
		"同一NFEのODEに対する改善幅は、10kで0.712、20kで0.377、50kで0.075、100kで0.012と、学習が進むにつれて縮小しました。既存モデルで見た「初期ほどRestartが効く」傾向が、均一な系列でも確認できました。",
		"100kでは通常ODE（35 NFE）に対してわずかに悪化（−0.003）しており、効果はほぼなくなっています。110k以降はFID-25kの比較がまだそろっていません。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 8. summary

pres.addSection({ title: "まとめ" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "まとめ" });
	s.addText("まとめと今後の確認", { placeholder: "title" });

	const lw = 5.9;
	chip(s, "わかったこと", MX, 1.25, 1.35);
	const findings = [
		[{ text: "既存モデル（FID-50k、3回評価）：", options: { bold: true } }, { text: "学習初期では改善（5.5k：+1.718）、後期では悪化（200k：−0.181）" }],
		[{ text: "再学習系列（FID-25k）：", options: { bold: true } }, { text: "同一NFEに対する改善幅は、10kの0.712から100kの0.012まで縮小" }],
		[{ text: "K探索（FID-5k）：", options: { bold: true } }, { text: "最良Kは学習段階によって一定ではない。改善幅は初期ほど大きい" }],
	];
	findings.forEach((runs, i) => {
		const y = 1.75 + i * 1.1;
		s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: MX, y, w: 0.42, h: 0.42, fill: { color: NAVY }, line: { type: "none" }, color: C.background1, fontSize: 15, bold: true, align: "center", valign: "middle", margin: 0, objectName: `finding ${i + 1}` });
		text(s, runs, { x: MX + 0.6, y: y + 0.03, w: lw - 0.6, h: 0.85, fontSize: 13.5 });
	});

	card(s, MX, 5.15, lw, 1.2, NAVY, "next step");
	chip(s, "今後", MX + 0.25, 5.33, 0.8, C.background1, NAVY);
	text(s, "K探索を25,000枚評価へ拡張し、学習段階ごとの最適Kを確認する", { x: MX + 0.25, y: 5.73, w: lw - 0.5, h: 0.5, fontSize: 14, bold: true, color: C.background1 });

	const rx = MX + lw + 0.4, rw = W - MX - rx;
	chip(s, "進捗（最新記録）", rx, 1.25, 1.75);
	const groups = [
		["評価済み", GREEN, [
			"既存モデルK=5：FID-50k × 3回（5.5k〜200kの4点）",
			"K探索：FID-5k（10k〜100k、K=0〜8）",
			"再学習系列K=5のFID-5k：10k〜120k",
			"再学習系列K=5のFID-25k：10k〜100k",
		]],
		["確認途中", ORANGE, [
			`再学習：10k〜${PROGRESS.savedLast}を保存済み、200kまで学習中`,
			"K探索のFID-25k：80kでK=0〜6まで評価済み",
			"再学習系列K=5のFID-25k：110k・120k",
		]],
		["未実施", MUTED, [
			"K探索のFID-25k：80k以外のcheckpoint",
			"再学習系列K=5のFID-50k、170k〜200kのRestart評価",
		]],
	];
	let gy = 1.75;
	groups.forEach(([label, color, items]) => {
		const gh = 0.5 + items.length * 0.31;
		card(s, rx, gy, rw, gh, TINT, `status ${label}`);
		chip(s, label, rx + 0.2, gy + 0.12, 1.05, color);
		text(s, items.map((item, i) => ({ text: item, options: { bullet: true, breakLine: i < items.length - 1 } })),
			{ x: rx + 0.2, y: gy + 0.5, w: rw - 0.35, h: items.length * 0.31, fontSize: 12, paraSpaceAfter: 2 });
		gy += gh + 0.15;
	});

	s.addNotes([
		"まとめです。既存モデルでは、Restartは学習初期で改善し、後期では同一NFEのODEより悪化しました。10k間隔の再学習系列でも、改善幅は10kの0.712から100kの0.012まで縮小し、同じ傾向を確認しました。",
		"K探索では最良Kが学習段階ごとに変わりましたが、5,000枚の評価なので候補選択の段階です。今後はK探索を25,000枚評価へ拡張し、最適Kを確認します。",
		"進捗：再学習は10k〜160kまで保存済みで200kまで学習中です。K探索のFID-25kは80kのK=0〜6まで評価済み、再学習系列K=5のFID-25kは110k・120kが途中です。K=5のFID-50kと、170k〜200kの評価は未実施です。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- write

// Theme fonts from pptxgenjs cover Latin text only; point Japanese text at the same font.
async function setEastAsianFont(file) {
	const fs = require("fs");
	const zip = await JSZip.loadAsync(fs.readFileSync(file));
	for (const name of Object.keys(zip.files).filter((n) => /^ppt\/theme\/theme\d+\.xml$/.test(n))) {
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
