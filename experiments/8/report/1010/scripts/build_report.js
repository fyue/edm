// Build the 1010 progress report deck (retraining and Restart condition checks).
//
// Usage (from the EDM repository root, after scripts/prepare_figures.py):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/1010/scripts/build_report.js
//
// Values on the slides come from the result TSVs of experiments 0-1, 0-2, 8-8 and 8-11;
// update the PROGRESS block and the tables when those change.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report", "1010");
const ASSETS = path.join(REPORT_DIR, "assets");
const OUT = path.join(REPORT_DIR, "restart_sampling_weekly_progress_report_claude.pptx");

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

// Latest records, checked when the deck is built.
const PROGRESS = {
	kSweepFid5k: "20k〜140k",
	kSweepFid25k: "20k〜80k",
	kSweepFid25kPartial: "100kはK=0〜3まで",
	subsetTrainedKimg: "約21.8k",
	subsetSaved: "10k・20k",
	subsetFid: "10k・20k",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "研究進捗";
const C = pres.SchemeColor;
const ORANGE = C.accent2, GREEN = C.accent4, NAVY = C.text2, TINT = C.background2;

const W = 13.333, MX = 0.55, CW = W - 2 * MX;

// ---------------------------------------------------------------------------- layouts

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

// The slide's message, in a full-width band above the footer.
function takeaway(slide, label, value, y) {
	const top = y || 6.3;
	card(slide, MX, top, CW, 0.58, NAVY, "takeaway");
	chip(slide, label, MX + 0.2, top + 0.14, 1.05, C.background1, NAVY);
	text(slide, value, { x: MX + 1.45, y: top + 0.08, w: CW - 1.65, h: 0.42, fontSize: 15, bold: true, color: C.background1, valign: "middle" });
}

function picture(slide, file, x, y, w, aspect, name) {
	slide.addImage({ path: path.join(ASSETS, file), x, y, w, h: w / aspect, objectName: name });
	return w / aspect;
}

function bullets(items) {
	return items.map((item, i) => ({ text: item, options: { bullet: true, breakLine: i < items.length - 1 } }));
}

// Fresh options per cell: pptxgenjs mutates option objects in place.
const head = (fontSize) => ({ bold: true, color: C.background1, fill: { color: NAVY }, align: "center", valign: "middle", fontSize });

// ---------------------------------------------------------------------------- 1. overview

pres.addSection({ title: "概要" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "概要" });
	s.addText("研究進捗", { placeholder: "title" });
	text(s, "CIFAR-10 class-conditional EDMにおける再学習とRestart条件の検証", { x: MX, y: 1.02, w: CW, h: 0.35, fontSize: 16, color: C.accent5 });
	chip(s, "今回の4実験", MX, 1.62, 1.4);

	const status = (label, color, detail) => [
		{ text: label, options: { bold: true, color, breakLine: true } },
		{ text: detail, options: { fontSize: 13 } },
	];
	const name = (value) => ({ text: value, options: { bold: true, color: NAVY } });
	const rows = [
		[{ text: "実験", options: head(14) }, { text: "目的", options: head(14) }, { text: "進捗", options: head(14) }],
		[name("10k間隔のcheckpoint作成"), "学習段階を均等な間隔で比較する",
			{ text: status("完了", GREEN, "200kまで再学習完了。全20点を保存") }],
		[name("Restart開始ノイズ量の比較"), "開始ノイズ量0.06と0.14の効果を比較する",
			{ text: status("完了", GREEN, "FID-50kまで評価完了") }],
		[name("開始ノイズ量0.14での最適K探索"), "学習段階ごとの最良Kを調べる",
			{ text: status("評価中", ORANGE, `FID-5kは${PROGRESS.kSweepFid5k}、FID-25kは${PROGRESS.kSweepFid25k}まで完了（${PROGRESS.kSweepFid25kPartial}）`) }],
		[name("30,000枚の学習データで再学習"), "データを減らした条件でFIDの推移を確認する",
			{ text: status("途中", ORANGE, `FID評価は20kまで完了（学習は${PROGRESS.subsetTrainedKimg}で停止中）`) }],
	];
	s.addTable(rows, {
		x: MX, y: 2.12, w: CW, colW: [3.75, 4.0, CW - 7.75], rowH: [0.45, 0.98, 0.98, 0.98, 0.98],
		fontSize: 14, color: C.text1, valign: "middle", margin: [0.06, 0.15, 0.06, 0.15],
		border: { type: "solid", pt: 1, color: "D9DCDF" }, fill: { color: THEME.colors.lt1 },
	});

	s.addNotes([
		"今回は4つの実験の進捗を報告します。",
		"1つ目の10k間隔のcheckpoint作成は、200kまで再学習が完了し、全20点を保存しました。2つ目のRestart開始ノイズ量の比較は、FID-50kまで評価が完了しています。",
		`3つ目の開始ノイズ量0.14での最適K探索は評価中で、FID-5kは${PROGRESS.kSweepFid5k}、FID-25kは${PROGRESS.kSweepFid25k}まで終わっています。4つ目の30,000枚の学習データでの再学習は、FID評価を20kまで行いました。学習は${PROGRESS.subsetTrainedKimg}で止めています。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 2. exact 10k checkpoints

pres.addSection({ title: "実験結果" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText("10k間隔のcheckpoint作成が完了", { placeholder: "title" });

	chip(s, "目的", MX, 1.3, 0.8);
	text(s, "不均一だったcheckpointを、10k間隔の系列に作り直す。", { x: MX, y: 1.75, w: 7.3, h: 0.5, fontSize: 16 });
	chip(s, "条件", MX, 2.6, 0.8);
	text(s, bullets([
		"CIFAR-10 class-conditional",
		"DDPM++（ネットワーク）、EDM（前処理・損失）",
		"学習率 0.001、EMA 0.5 Mimg",
		"学習seed = 0",
	]), { x: MX, y: 3.05, w: 7.3, h: 2.4, fontSize: 16, paraSpaceAfter: 12 });

	const px = 8.35, pw = W - MX - px, py = 1.3, ph = 4.7;
	card(s, px, py, pw, ph, TINT, "result");
	chip(s, "結果", px + 0.35, py + 0.35, 0.8);
	text(s, [
		{ text: "20", options: { fontSize: 66, bold: true, color: NAVY } },
		{ text: " / 20点", options: { fontSize: 24, color: NAVY } },
	], { x: px + 0.35, y: py + 0.9, w: pw - 0.7, h: 1.25, valign: "bottom" });
	text(s, "10k〜200kの全checkpointを保存", { x: px + 0.35, y: py + 2.3, w: pw - 0.7, h: 0.4, fontSize: 17, bold: true, color: NAVY });
	text(s, "200kまで再学習を完了", { x: px + 0.35, y: py + 2.9, w: pw - 0.7, h: 0.35, fontSize: 15 });
	text(s, "各保存点は、指定の学習量に到達した最初の更新の直後に保存（batch 512のため超過は最大384枚）", { x: px + 0.35, y: py + 3.6, w: pw - 0.7, h: 0.8, fontSize: 12, color: MUTED });

	takeaway(s, "ポイント", "学習段階を均等な間隔で比較できる系列が揃った");
	s.addNotes([
		"前回報告した10k間隔のcheckpoint作成が完了しました。既存のcheckpointは間隔が不均一だったので、同じ学習条件で再学習し、10kごとに保存しています。",
		"条件はDDPM++・EDM、学習率0.001、EMA 0.5 Mimg、学習seed 0です。200kまで再学習を完了し、10kから200kまでの全20点を保存しました。",
		"これで、学習段階を均等な間隔で比較できる系列が揃いました。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 3. t_min comparison

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText("Restart開始ノイズ量の比較", { placeholder: "title" });

	const y = 1.2, h = 1.3, lw = 7.6;
	card(s, MX, y, lw, h, TINT, "setup");
	chip(s, "条件", MX + 0.2, y + 0.15, 0.8);
	text(s, bullets([
		"K=5（55 NFE）、終了側のノイズ量 t_max = 0.30 を固定",
		"開始側のノイズ量 t_min を 0.06 / 0.14 で比較",
		"各条件50,000枚（FID-50k、seed 0–49,999）",
	]), { x: MX + 1.2, y: y + 0.17, w: lw - 1.4, h: 1.0, fontSize: 13, paraSpaceAfter: 3 });

	const rx = MX + lw + 0.25, rw = W - MX - rx;
	card(s, rx, y, rw, h, TINT, "t_min");
	chip(s, "t_minとは", rx + 0.2, y + 0.15, 1.15);
	text(s, "ノイズを加え直す直前のノイズ量。Restart区間 [t_min, t_max] の下端", { x: rx + 0.2, y: y + 0.58, w: rw - 0.4, h: 0.65, fontSize: 13 });

	const fw = 8.1;
	const fh = picture(s, "tmin_comparison_fid50k.png", MX, 2.7, fw, 1620 / 593, "figure t_min FID-50k");
	text(s, "左：t_minごとのFID-50k（低いほど良い）。右：FID(0.06) − FID(0.14)（正値なら0.14が良い）。",
		{ x: MX, y: 2.7 + fh + 0.05, w: fw, h: 0.3, fontSize: 10.5, color: MUTED });

	const tx = MX + fw + 0.35, tw = W - MX - tx;
	text(s, "FID-50k（低いほど良い）", { x: tx, y: 2.75, w: tw, h: 0.3, fontSize: 12.5, bold: true, color: NAVY });
	const better = () => ({ bold: true, color: GREEN });
	const tRows = [
		[{ text: "checkpoint", options: head(12) }, { text: "t_min 0.06", options: head(12) }, { text: "t_min 0.14", options: head(12) }],
		["20k", { text: "2.6230", options: better() }, "2.7131"],
		["40k", { text: "2.1460", options: better() }, "2.1697"],
		["60k", "2.0402", { text: "2.0303", options: better() }],
		["170k*", "1.8781", { text: "1.8402", options: better() }],
	];
	s.addTable(tRows, {
		x: tx, y: 3.1, w: tw, colW: [1.2, (tw - 1.2) / 2, (tw - 1.2) / 2], rowH: 0.36,
		fontSize: 12.5, color: C.text1, align: "center", valign: "middle", margin: [0.02, 0.05, 0.02, 0.05],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});
	text(s, "緑字：良い方の t_min\n*170kは別の学習系列（160kから学習率2e-4で追加学習した最良モデル）", { x: tx, y: 5.0, w: tw, h: 0.9, fontSize: 10.5, color: MUTED });

	takeaway(s, "ポイント", "開始ノイズ量の良し悪しは、モデルによって変わる");
	s.addNotes([
		"次に、Restartの開始ノイズ量の比較です。t_minは、ノイズを加え直す直前のノイズ量で、Restart区間の下端にあたります。K=5、55 NFE、終了側のノイズ量0.30を固定して、開始側を0.06と0.14で比べました。0.14はRestart論文の公式EDM設定の値です。",
		"FID-50kでは、20kと40kでは0.06、60kと170kでは0.14の方が良い結果でした。170kでは1.8781から1.8402へ改善しています。",
		"なお、170kは今回の10k間隔系列ではなく、160kから学習率2e-4で追加学習した別系列の最良モデルです。",
		"開始ノイズ量の良し悪しは、モデルによって変わるという結果です。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 4-5. K sweep with t_min = 0.14

function kSweepSlide(stage, figure, status, ckptCells, kCells, gainCells, result, notes) {
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText(`開始ノイズ量0.14で最適なRestart回数を探索（${stage}）`, { placeholder: "title" });

	chip(s, "条件", MX, 1.13, 0.8);
	text(s, `主経路18 steps、開始ノイズ量 t_min = 0.14、K=0〜8、NFE = 35+4K（35〜67）、${stage}`, { x: MX + 0.95, y: 1.15, w: CW - 0.95, h: 0.3, fontSize: 13 });
	chip(s, "評価途中", MX, 1.53, 0.8, ORANGE);
	text(s, status, { x: MX + 0.95, y: 1.55, w: CW - 0.95, h: 0.3, fontSize: 13 });

	const fw = 11.6;
	const fh = picture(s, figure, (W - fw) / 2, 1.95, fw, 3240 / 900, `figure K sweep ${stage}`);
	text(s, `左：Kごとの${stage}（低いほど良い）。中央：FID(K) − FID(K=0)（負値ほどRestartで改善）。右：各checkpointの最良K。`,
		{ x: MX, y: 1.95 + fh + 0.04, w: CW, h: 0.28, fontSize: 10.5, color: MUTED });

	const label = (value) => ({ text: value, options: { bold: true, color: NAVY, align: "left" } });
	const rows = [
		[{ text: "checkpoint", options: head(11) }].concat(ckptCells.map((c) => ({ text: c, options: head(11) }))),
		[label("最良K")].concat(kCells),
		[label("改善幅（K=0比）")].concat(gainCells),
	];
	const firstW = 1.9;
	s.addTable(rows, {
		x: MX, y: 5.52, w: CW, colW: [firstW].concat(ckptCells.map(() => (CW - firstW) / ckptCells.length)), rowH: 0.23,
		fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.05, 0.01, 0.05],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});

	takeaway(s, result[0], result[1], 6.35);
	s.addNotes(notes.join("\n\n"));
}

const gain = (v) => ({ text: v, options: { color: GREEN, bold: true } });
const pending = () => ({ text: "評価途中", options: { color: ORANGE } });

kSweepSlide("FID-5k", "fixed_main18_tmin014_fid5000.png",
	`${PROGRESS.kSweepFid5k}まで評価済み（160k〜200kは未評価）`,
	["20k", "40k", "60k", "80k", "100k", "120k", "140k"],
	["K=5", "K=5", "K=2", "K=2", "K=2", "K=1", "K=1"],
	["+0.333", "+0.213", "+0.184", "+0.124", "+0.105", "+0.052", "+0.102"].map(gain),
	["結果", "20k・40kはK=5、60k〜100kはK=2、120k・140kはK=1が最良"],
	[
		"開始ノイズ量を0.14に固定して、最適なRestart回数Kを探索しました。主経路は18 stepsで、Kを0から8まで振っています。NFEは35+4Kです。",
		"まずFID-5kの結果です。左がKごとのFID、中央がK=0との差、右が各checkpointの最良Kです。",
		"20kと40kではK=5、60kから100kではK=2、120kと140kではK=1が最良で、学習が進むほど最良Kが小さくなっています。",
		`FID-5kは${PROGRESS.kSweepFid5k}まで評価済みで、160k以降は未評価です。改善幅はFID(K=0) − 最小FIDです。`,
	]);

kSweepSlide("FID-25k", "fixed_main18_tmin014_fid25000.png",
	`${PROGRESS.kSweepFid25k}は全Kを評価済み、${PROGRESS.kSweepFid25kPartial}評価（120k以降は未評価）`,
	["20k", "40k", "60k", "80k", "100k"],
	["K=7", "K=7", "K=3", "K=2", pending()],
	["+0.390", "+0.143", "+0.096", "+0.081"].map(gain).concat([pending()]),
	["ポイント", "最良Kは学習段階と評価画像数によって変わるため、評価を増やして確認する"],
	[
		"同じ条件をFID-25kで評価した結果です。全Kが揃った範囲では、20kと40kでK=7、60kでK=3、80kでK=2が最良でした。",
		`100kはK=0〜3までの途中結果なので、まだ最良Kは決めていません。FID-25kでも、学習が進むほど最良Kが小さくなる傾向は同じですが、20k・40kではFID-5kのK=5からK=7に変わっています。`,
		"最良Kは学習段階と評価画像数によって変わるので、評価を増やして確認します。",
	]);

// ---------------------------------------------------------------------------- 6. 30k subset

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "実験結果" });
	s.addText("学習データ30,000枚での再学習", { placeholder: "title" });

	const y = 1.2, lw = 7.6, h = 2.15;
	card(s, MX, y, lw, h, TINT, "setup");
	chip(s, "条件", MX + 0.2, y + 0.15, 0.8);
	text(s, bullets([
		"CIFAR-10の50,000枚から、固定seedで30,000枚を抽出",
		"モデル（DDPM++・EDM）、学習率0.001、EMA 0.5 Mimgは全データ学習と同じ",
	]), { x: MX + 0.2, y: y + 0.55, w: lw - 0.4, h: 0.8, fontSize: 13.5, paraSpaceAfter: 4 });
	chip(s, "評価", MX + 0.2, y + 1.38, 0.8);
	text(s, "通常ODE、18 steps（35 NFE）、FID-50k（seed 0–49,999）", { x: MX + 1.2, y: y + 1.4, w: lw - 1.4, h: 0.3, fontSize: 13.5 });

	const rx = MX + lw + 0.25, rw = W - MX - rx;
	card(s, rx, y, rw, h, TINT, "progress");
	chip(s, "進捗", rx + 0.2, y + 0.15, 0.8, ORANGE);
	text(s, [
		{ text: "学習の到達点", options: { bold: true, color: NAVY, breakLine: true } },
		{ text: `${PROGRESS.subsetTrainedKimg}（保存済み：${PROGRESS.subsetSaved}）で停止中`, options: { breakLine: true } },
		{ text: "FID評価済み", options: { bold: true, color: NAVY, breakLine: true } },
		{ text: PROGRESS.subsetFid },
	], { x: rx + 0.2, y: y + 0.58, w: rw - 0.4, h: 1.45, fontSize: 13.5, paraSpaceAfter: 3 });

	// FID-50k by checkpoint.
	const sy = 3.6, sh = 2.45;
	card(s, MX, sy, CW, sh, TINT, "fid");
	text(s, "FID-50k（通常ODE、35 NFE、低いほど良い）", { x: MX + 0.3, y: sy + 0.2, w: 6, h: 0.3, fontSize: 13, bold: true, color: NAVY });
	[["10k", "5.4234", 2.3], ["20k", "2.8179", 7.4]].forEach(([ckpt, fid, x]) => {
		text(s, fid, { x, y: sy + 0.65, w: 3.6, h: 1.0, fontSize: 54, bold: true, color: NAVY, align: "center", valign: "middle" });
		text(s, `checkpoint ${ckpt}`, { x, y: sy + 1.7, w: 3.6, h: 0.35, fontSize: 14, color: MUTED, align: "center" });
	});
	s.addShape(pres.shapes.RIGHT_ARROW, { x: 6.35, y: sy + 0.9, w: 0.65, h: 0.55, fill: { color: C.accent5 }, line: { type: "none" }, objectName: "fid arrow" });

	takeaway(s, "ポイント", "20kまでFIDは改善した。データ削減の影響は今後の比較で確認する");
	s.addNotes([
		"最後に、学習データを30,000枚に減らした再学習です。CIFAR-10の50,000枚から固定seedで30,000枚を抽出し、モデル、学習率、EMAは全データ学習と同じにしています。",
		"評価は通常ODE、18 steps、35 NFEのFID-50kです。10kで5.4234、20kで2.8179と、20kまでFIDは改善しました。",
		`学習は${PROGRESS.subsetTrainedKimg}まで進めて停止しており、保存済みのcheckpointは${PROGRESS.subsetSaved}です。FID評価は${PROGRESS.subsetFid}まで終わっています。データ削減の影響は、今後全データ学習と比較して確認します。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 7. summary

pres.addSection({ title: "まとめ" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "まとめ" });
	s.addText("まとめと今後の確認", { placeholder: "title" });

	const groups = [
		{ label: "わかったこと", color: GREEN, x: MX, items: [
			["10k間隔のcheckpoint系列が完成した", "10k〜200kの20点。学習段階を均等な間隔で比較できる"],
			["Restartの開始ノイズ量と回数は、モデルに合わせて確認する必要がある", "t_minは学習段階で良い値が変わり、最良Kも学習段階と評価画像数で変化した"],
		] },
		{ label: "今後の確認", color: NAVY, x: 7.03, items: [
			["K探索の残りのcheckpointと、より多い画像数での評価を進める", "FID-5kは160k〜200k、FID-25kは100k以降が残り"],
			["30,000枚学習を継続評価し、全データ学習と同じ条件で比較する", "同じcheckpoint・同じ評価条件（通常ODE、FID-50k）でそろえる"],
		] },
	];
	const colW = 5.75, cardH = 2.15, cardY = [1.85, 4.2];
	let n = 0;
	groups.forEach(({ label, color, x, items }) => {
		chip(s, label, x, 1.3, 1.4, color);
		items.forEach(([title, desc], i) => {
			n += 1;
			const cy = cardY[i];
			card(s, x, cy, colW, cardH, TINT, `${label} ${n}`);
			s.addText(String(n), { shape: pres.shapes.OVAL, x: x + 0.3, y: cy + 0.3, w: 0.5, h: 0.5, fill: { color }, line: { type: "none" }, color: C.background1, fontSize: 17, bold: true, align: "center", valign: "middle", margin: 0, objectName: `number ${n}` });
			text(s, title, { x: x + 1.0, y: cy + 0.3, w: colW - 1.25, h: 0.85, fontSize: 16.5, bold: true, color: NAVY });
			text(s, desc, { x: x + 1.0, y: cy + 1.1, w: colW - 1.25, h: 0.75, fontSize: 13.5 });
		});
	});
	s.addShape(pres.shapes.RIGHT_ARROW, { x: MX + colW + 0.12, y: 3.85, w: 0.48, h: 0.55, fill: { color: C.accent5 }, line: { type: "none" }, objectName: "flow arrow" });

	s.addNotes([
		"まとめです。10k間隔のcheckpoint系列が完成しました。Restartについては、開始ノイズ量も回数も、モデルの学習段階によって良い値が変わるので、モデルに合わせて確認する必要があります。",
		"今後は、K探索の残りのcheckpointと、より多い画像数での評価を進めます。また、30,000枚学習を継続評価し、全データ学習と同じ条件で比較します。",
	].join("\n\n"));
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
