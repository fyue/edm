// Build the 1017 progress report deck (Restart K sweeps with t_min = 0.14).
//
// Usage (from the EDM repository root, after scripts/prepare_figures.py):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/1017/scripts/build_report.js
//
// Best K, gains and progress are computed from the snapshot in assets/. The sentences that
// describe them are written by hand, so the build stops if the snapshot no longer matches.

const fs = require("fs");
const path = require("path");
const assert = require("assert");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report", "1017");
const ASSETS = path.join(REPORT_DIR, "assets");
const OUT = path.join(REPORT_DIR, "restart_k_sweep_progress_report_claude.pptx");

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

// ---------------------------------------------------------------------------- data

// fid_by_checkpoint_and_k.tsv -> { stage: { ckpt: { k: fid } } }, checkpoints in training order.
function readSweep(file) {
	const rows = fs.readFileSync(path.join(ASSETS, file), "utf8").trim().split("\n").slice(1);
	const out = {};
	for (const row of rows) {
		const [ckpt, stage, k, fid] = row.split("\t");
		((out[stage] = out[stage] || {})[ckpt] = out[stage][ckpt] || {})[Number(k)] = Number(fid);
	}
	return out;
}

// Best K and gain over K=0 per checkpoint, for checkpoints with all nine K evaluated.
function bestK(stageTable) {
	return Object.keys(stageTable)
		.sort((a, b) => parseInt(a, 10) - parseInt(b, 10))
		.filter((ckpt) => Object.keys(stageTable[ckpt]).length === 9)
		.map((ckpt) => {
			const fids = stageTable[ckpt];
			const k = Object.keys(fids).map(Number).reduce((a, b) => (fids[b] < fids[a] ? b : a));
			return { ckpt, k, gain: fids[0] - fids[k] };
		});
}

const MAIN18 = readSweep("fixed_main18__fid_by_checkpoint_and_k.tsv");
const NFE55 = readSweep("fixed_nfe55__fid_by_checkpoint_and_k.tsv");
const PROGRESS = JSON.parse(fs.readFileSync(path.join(ASSETS, "progress.json"), "utf8"));

const main18Fid50k = bestK(MAIN18["50000"]);
const main18Fid25k = bestK(MAIN18["25000"]);
const nfe55Fid25k = bestK(NFE55["25000"]);
const nfe55Fid50kDone = PROGRESS.fixed_nfe55["50000"] || [];
const nfe55Total = PROGRESS.fixed_nfe55["5000"].length;

// The hand-written sentences below describe exactly these results.
assert.deepStrictEqual(main18Fid50k.map((r) => r.k), [7, 3, 3, 3, 3, 3, 1, 3, 1, 1]);
assert.deepStrictEqual(nfe55Fid25k.map((r) => r.k), [8, 8, 4, 4, 4, 4]);
assert.deepStrictEqual(main18Fid25k.slice(0, 6).map((r) => r.k), [7, 7, 3, 2, 2, 3]);
assert.ok(nfe55Fid25k.every((r) => r.gain > 0));
assert.strictEqual(PROGRESS.fixed_main18["50000"].length, 90);
assert.strictEqual(PROGRESS.fixed_nfe55["25000"].length, nfe55Total);

const fmtGain = (g) => `${g >= 0 ? "+" : "−"}${Math.abs(g).toFixed(3)}`;
const firstGain = fmtGain(main18Fid50k[0].gain), lastGain = fmtGain(main18Fid50k[main18Fid50k.length - 1].gain);

// "20kのK=0〜4" for the conditions done at one checkpoint.
function doneSummary(conds) {
	const byCkpt = {};
	conds.forEach((c) => { const [ckpt, k] = c.split("-k"); (byCkpt[ckpt] = byCkpt[ckpt] || []).push(Number(k)); });
	return Object.keys(byCkpt).sort((a, b) => parseInt(a, 10) - parseInt(b, 10)).map((ckpt) => {
		const ks = byCkpt[ckpt].sort((a, b) => a - b);
		return ks.length === 9 ? `${ckpt}は全K` : `${ckpt}のK=${ks[0]}〜${ks[ks.length - 1]}`;
	}).join("、");
}
const nfe55Fid50kText = `FID-50kは${doneSummary(nfe55Fid50kDone)}まで（${nfe55Fid50kDone.length} / ${nfe55Total}条件）`;
const nfe55Remaining = nfe55Total - nfe55Fid50kDone.length;

// ---------------------------------------------------------------------------- deck

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "研究進捗";
const C = pres.SchemeColor;
const ORANGE = C.accent2, GREEN = C.accent4, NAVY = C.text2, TINT = C.background2;

const W = 13.333, MX = 0.55, CW = W - 2 * MX;

pres.defineSlideMaster({
	title: "CONTENT",
	background: { color: THEME.colors.lt1 },
	objects: [
		{ placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: CW, h: 0.7, fontSize: 26, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0 }, text: "" } },
	],
	slideNumber: { x: 12.33, y: 7.02, w: 0.45, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

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

// Fresh options per cell: pptxgenjs mutates option objects in place.
const head = (fontSize) => ({ bold: true, color: C.background1, fill: { color: NAVY }, align: "center", valign: "middle", fontSize });
const rowLabel = () => ({ bold: true, color: NAVY, align: "left" });
const gainCell = (g) => ({ text: fmtGain(g), options: { bold: true, color: g > 0 ? GREEN : C.accent6 } });

// Best-K / gain table under a result figure.
function bestKTable(slide, results, gainLabel, y) {
	const firstW = 2.6, colW = (CW - firstW) / results.length;
	const rows = [
		[{ text: "checkpoint", options: head(11) }].concat(results.map((r) => ({ text: r.ckpt, options: head(11) }))),
		[{ text: "最良K", options: rowLabel() }].concat(results.map((r) => `K=${r.k}`)),
		[{ text: gainLabel, options: rowLabel() }].concat(results.map((r) => gainCell(r.gain))),
	];
	slide.addTable(rows, {
		x: MX, y, w: CW, colW: [firstW].concat(results.map(() => colW)), rowH: 0.25,
		fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.05, 0.01, 0.05],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});
}

const FIG_W = 11.6, FIG_ASPECT = 3240 / 900;

// ---------------------------------------------------------------------------- 1. overview

pres.addSection({ title: "概要" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "概要" });
	s.addText("研究進捗：Restart回数Kの探索", { placeholder: "title" });
	text(s, "開始ノイズ量 t_min = 0.14、10k間隔で再学習したcheckpoint系列、K = 0〜8", { x: MX, y: 1.02, w: CW, h: 0.35, fontSize: 16, color: C.accent5 });

	const status = (label, color, detail) => ({ text: [
		{ text: label, options: { bold: true, color, breakLine: true } },
		{ text: detail, options: { fontSize: 13 } },
	] });
	const rows = [
		[{ text: "", options: head(14) }, { text: "主経路を固定", options: head(15) }, { text: "計算量を固定", options: head(15) }],
		[{ text: "考え方", options: rowLabel() }, "主経路はそのまま。Restartを足すほど計算量が増える", "計算量をそろえ、主経路を減らした分をRestartに回す"],
		[{ text: "主経路", options: rowLabel() }, "18 steps（全K共通）", "28−2K steps（K=0：28 → K=8：12）"],
		[{ text: "NFE", options: rowLabel() }, "35+4K（35〜67）", "55（全K共通）"],
		[{ text: "K=0（Restartなし）", options: rowLabel() }, "通常ODE 18 steps（35 NFE）", "通常ODE 28 steps（55 NFE）"],
		[{ text: "checkpoint", options: rowLabel() }, `20k〜200k（${main18Fid50k.length}点）`, `20k〜120k（${nfe55Fid25k.length}点）`],
		[{ text: "進捗", options: rowLabel() },
			status("完了", GREEN, `FID-5k・25k・50kとも全${PROGRESS.fixed_main18["50000"].length}条件`),
			status("評価中", ORANGE, `FID-5k・25kは全${nfe55Total}条件、${nfe55Fid50kText}`)],
	];
	s.addTable(rows, {
		x: MX, y: 1.65, w: CW, colW: [2.5, (CW - 2.5) / 2, (CW - 2.5) / 2], rowH: [0.45, 0.62, 0.55, 0.55, 0.55, 0.55, 0.85],
		fontSize: 14, color: C.text1, valign: "middle", margin: [0.05, 0.15, 0.05, 0.15],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});

	s.addNotes([
		"今回は、開始ノイズ量t_minを0.14に固定して、Restart回数Kを0から8まで振った2つの実験の進捗を報告します。どちらも10k間隔で再学習したcheckpoint系列を使っています。",
		"1つ目は主経路を18 stepsに固定した探索で、Kを増やすほど計算量も増えます。NFEは35+4Kです。2つ目は計算量を55 NFEに固定した探索で、Kを1増やすごとに主経路を2 steps減らし、その分をRestartに回します。こちらのK=0は28 stepsの通常ODEなので、同じ55 NFEのODEとの比較になります。",
		`主経路固定の方は20k〜200kの全条件でFID-50kまで完了しました。計算量固定の方は20k〜120kでFID-25kまで完了し、FID-50kは評価中です。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 2. fixed main path

pres.addSection({ title: "結果" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
	s.addText("主経路を固定したK探索（FID-50k）", { placeholder: "title" });
	chip(s, "条件", MX, 1.13, 0.8);
	text(s, "主経路18 steps、K=0〜8、NFE = 35+4K（35〜67）、t_min = 0.14、各条件50,000枚、20k〜200k", { x: MX + 0.95, y: 1.15, w: CW - 0.95, h: 0.3, fontSize: 13 });

	const fh = picture(s, "fixed_main18__fixed_main18_tmin014_fid50000.png", (W - FIG_W) / 2, 1.6, FIG_W, FIG_ASPECT, "figure fixed main path FID-50k");
	text(s, "左：KごとのFID（低いほど良い）。中央：FID(K) − FID(K=0)（青ほど改善、赤ほど悪化）。右：各checkpointの最良K。",
		{ x: MX, y: 1.6 + fh + 0.04, w: CW, h: 0.28, fontSize: 10.5, color: MUTED });
	bestKTable(s, main18Fid50k, "改善幅（K=0比）", 5.28);

	takeaway(s, "ポイント", `最良Kは20kでK=7、40k〜120kでK=3、後期はK=1〜3。改善幅は${firstGain}から${lastGain}へ縮小`);
	s.addNotes([
		"主経路を18 stepsに固定した探索の、FID-50kの結果です。左がKごとのFID、中央がK=0との差、右が各checkpointの最良Kです。",
		`最良Kは、20kでK=7、40kから120kではK=3で安定し、140k以降はK=1〜3です。K=0に対する改善幅は20kの${firstGain}から200kの${lastGain}まで小さくなりました。`,
		"中央の図では、学習後期でKを大きくすると赤、つまりK=0より悪化しています。なお、この条件ではKを増やすと計算量も増えるので、改善幅には計算量の増加分も含まれます。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 3. fixed compute

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
	s.addText("計算量を固定したK探索（FID-25k）", { placeholder: "title" });
	chip(s, "条件", MX, 1.13, 0.8);
	text(s, "NFE = 55で固定、主経路28−2K steps、K=0〜8、t_min = 0.14、各条件25,000枚、20k〜120k", { x: MX + 0.95, y: 1.15, w: CW - 0.95, h: 0.3, fontSize: 13 });
	chip(s, "評価途中", MX, 1.53, 0.8, ORANGE);
	text(s, `${nfe55Fid50kText}。FID-50kの図はまだ点が少ないため、FID-25kの結果を示す`, { x: MX + 0.95, y: 1.55, w: CW - 0.95, h: 0.3, fontSize: 13 });

	const figW = 11.2;
	const fh = picture(s, "fixed_nfe55__fixed_nfe55_tmin014_fid25000.png", (W - figW) / 2, 1.98, figW, FIG_ASPECT, "figure fixed compute FID-25k");
	text(s, "左：KごとのFID（低いほど良い）。中央：FID(K) − FID(K=0)（青ほど改善、赤ほど悪化）。右：各checkpointの最良K。K=0は同じ55 NFEの通常ODE。",
		{ x: MX, y: 1.98 + fh + 0.04, w: CW, h: 0.28, fontSize: 10.5, color: MUTED });
	bestKTable(s, nfe55Fid25k, "改善幅（K=0比）", 5.47);

	takeaway(s, "ポイント", "同じ55 NFEのODEより全checkpointで改善。最良Kは20k・40kでK=8、60k〜120kでK=4", 6.35);
	s.addNotes([
		"計算量を55 NFEに固定した探索の、FID-25kの結果です。K=0は28 stepsの通常ODEなので、ここでの改善幅は同じ計算量のODEに対する改善になります。",
		`すべてのcheckpointでRestartの方が良く、改善幅は20kで${fmtGain(nfe55Fid25k[0].gain)}、120kで${fmtGain(nfe55Fid25k[nfe55Fid25k.length - 1].gain)}でした。最良Kは20kと40kでK=8、60k以降はK=4です。`,
		`FID-50kは評価中で、現在${nfe55Fid50kDone.length}条件が終わっています。1条件およそ1時間かかるので、残り${nfe55Remaining}条件で約2日の見込みです。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 4. actual t_min per K

{
	const TMIN = JSON.parse(fs.readFileSync(path.join(ASSETS, "actual_tmin.json"), "utf8"));
	const main18 = TMIN.fixed_main18, nfe55 = TMIN.fixed_nfe55;
	const ks = [...Array(9).keys()];
	const even = (k) => k > 0 && k % 2 === 0;
	// The sentences below describe exactly these values.
	assert.ok(main18.slice(1).every((r) => r.actual_t_min === main18[1].actual_t_min));
	assert.ok(Math.max(...ks.filter(even).map((k) => nfe55[k].actual_t_min)) < Math.min(...ks.filter((k) => k % 2 === 1).map((k) => nfe55[k].actual_t_min)));
	assert.ok(nfe55Fid25k.slice(0, 2).every((r) => r.k === 8) && nfe55Fid25k.slice(2).every((r) => r.k === 4));
	const evenTmins = ks.filter(even).map((k) => nfe55[k].actual_t_min);
	const tminRange = `${Math.min(...evenTmins).toFixed(3)}〜${Math.max(...evenTmins).toFixed(3)}`;

	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "結果" });
	s.addText("Kごとの実際の開始ノイズ量（actual t_min）", { placeholder: "title" });
	text(s, "Restartは、指定した t_min = 0.14 に最も近い主経路スケジュール上のノイズ量から始まる。\n主経路のstep数が変わると、実際の開始ノイズ量も変わる。", { x: MX, y: 1.12, w: CW, h: 0.7, fontSize: 14 });

	const tmin = (r) => (r.actual_t_min === null ? "—" : r.actual_t_min.toFixed(4));
	const group = (label) => ({ text: label, options: { rowspan: 3, bold: true, color: NAVY, align: "center", valign: "middle", fill: { color: THEME.colors.lt2 } } });
	const item = (label) => ({ text: label, options: rowLabel() });
	const evenCell = (value) => ({ text: value, options: { bold: true, color: ORANGE, fill: { color: "FBE3D3" } } });
	const rows = [
		[{ text: "", options: head(13) }, { text: "", options: head(13) }].concat(ks.map((k) => ({ text: `K=${k}`, options: head(13) }))),
		[group("主経路を固定"), item("主経路 steps")].concat(main18.map((r) => String(r.main_steps))),
		[item("NFE")].concat(main18.map((r) => String(r.nfe))),
		[item("actual t_min")].concat(main18.map(tmin)),
		[group("計算量を固定"), item("主経路 steps")].concat(nfe55.map((r) => String(r.main_steps))),
		[item("NFE")].concat(nfe55.map((r) => String(r.nfe))),
		[item("actual t_min")].concat(nfe55.map((r, k) => (even(k) ? evenCell(tmin(r)) : tmin(r)))),
	];
	const groupW = 1.75, itemW = 1.65, kW = (CW - groupW - itemW) / 9;
	s.addTable(rows, {
		x: MX, y: 2.0, w: CW, colW: [groupW, itemW].concat(ks.map(() => kW)), rowH: 0.42,
		fontSize: 13, color: C.text1, align: "center", valign: "middle", margin: [0.02, 0.06, 0.02, 0.06],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});
	text(s, "K=0はRestartなし（—）。色付きのセル：計算量固定で実際の開始ノイズ量が小さい偶数K。", { x: MX, y: 5.02, w: CW, h: 0.28, fontSize: 11, color: MUTED });

	card(s, MX, 5.42, CW, 0.66, TINT, "caution");
	chip(s, "注意", MX + 0.2, 5.6, 0.8, ORANGE);
	text(s, "計算量固定の最良K（8, 4）はどちらも偶数。Kの効果と開始ノイズ量の効果が混ざっている可能性がある", { x: MX + 1.2, y: 5.6, w: CW - 1.4, h: 0.3, fontSize: 14 });

	takeaway(s, "ポイント", `計算量固定では、偶数Kで実際の開始ノイズ量が小さくなる（${tminRange}）`, 6.3);

	const zigzag = (ckpt) => ks.map((k) => NFE55["25000"][ckpt][k].toFixed(3)).join(", ");
	s.addNotes([
		`Restartはt_min=0.14を指定しても、主経路のノイズスケジュール上で0.14に最も近い点から始まります。主経路を18 stepsに固定した探索では、K=1〜8のすべてで${main18[1].actual_t_min.toFixed(4)}と同じです。`,
		`一方、計算量を固定した探索では主経路が28−2K stepsと変わるため、実際の開始ノイズ量もKごとに変わります。奇数Kでは0.13〜0.15ですが、偶数Kでは${tminRange}と小さくなります。`,
		`FID-25kでも、20kではK=0〜8の順に ${zigzag("20k")}、40kでは ${zigzag("40k")} と、偶数KでFIDが低いジグザグが見られます。最良Kの8と4はどちらも偶数なので、Kを増やした効果と、開始ノイズ量が小さくなった効果が混ざっている可能性があります。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 5. summary

pres.addSection({ title: "まとめ" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "まとめ" });
	s.addText("まとめと今後", { placeholder: "title" });

	chip(s, "最良Kの比較（FID-25k、20k〜120k）", MX, 1.25, 4.3);
	const cmp = main18Fid25k.slice(0, nfe55Fid25k.length);
	const rows = [
		[{ text: "checkpoint", options: head(12) }].concat(nfe55Fid25k.map((r) => ({ text: r.ckpt, options: head(12) }))),
		[{ text: "主経路を固定", options: rowLabel() }].concat(cmp.map((r) => `K=${r.k}`)),
		[{ text: "計算量を固定", options: rowLabel() }].concat(nfe55Fid25k.map((r) => ({ text: `K=${r.k}`, options: { bold: true, color: NAVY } }))),
	];
	const firstW = 2.6;
	s.addTable(rows, {
		x: MX, y: 1.7, w: CW, colW: [firstW].concat(nfe55Fid25k.map(() => (CW - firstW) / nfe55Fid25k.length)), rowH: 0.36,
		fontSize: 13, color: C.text1, align: "center", valign: "middle", margin: [0.02, 0.05, 0.02, 0.05],
		border: { type: "solid", pt: 1, color: "D9DCDF" },
	});

	const y = 3.15, h = 3.3, lw = 7.4;
	card(s, MX, y, lw, h, TINT, "findings");
	chip(s, "わかったこと", MX + 0.25, y + 0.2, 1.35, GREEN);
	const findings = [
		["主経路を固定：", "学習が進むと最良Kは小さくなり、改善幅も縮小した（FID-50k）"],
		["計算量を固定：", "同じ55 NFEのODEより全checkpointで改善。最良Kは主経路固定より大きめ（FID-25k）"],
	];
	findings.forEach(([label, body], i) => {
		const iy = y + 0.8 + i * 1.2;
		s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: MX + 0.25, y: iy, w: 0.42, h: 0.42, fill: { color: NAVY }, line: { type: "none" }, color: C.background1, fontSize: 15, bold: true, align: "center", valign: "middle", margin: 0, objectName: `finding ${i + 1}` });
		text(s, [{ text: label, options: { bold: true, color: NAVY } }, { text: body }], { x: MX + 0.85, y: iy + 0.02, w: lw - 1.1, h: 0.9, fontSize: 14.5 });
	});

	const rx = MX + lw + 0.25, rw = W - MX - rx;
	card(s, rx, y, rw, h, NAVY, "next");
	chip(s, "今後", rx + 0.25, y + 0.2, 0.8, C.background1, NAVY);
	text(s, [
		{ text: "計算量固定のFID-50kを完了する", options: { bold: true, breakLine: true } },
		{ text: `残り${nfe55Remaining}条件、約2日`, options: { fontSize: 13, color: C.background2, breakLine: true } },
		{ text: "両条件をFID-50kで比較する", options: { bold: true } },
	], { x: rx + 0.25, y: y + 0.75, w: rw - 0.5, h: 2.0, fontSize: 15, color: C.background1, paraSpaceAfter: 10 });

	s.addNotes([
		"まとめです。FID-25kで両条件の最良Kを並べると、計算量を固定した方が大きいKを選んでいます。主経路を減らしてでもRestartを増やした方が良い、という結果です。",
		"主経路固定では、学習が進むと最良Kが小さくなり、改善幅も縮小しました。計算量固定では、同じ55 NFEのODEより全checkpointで改善しています。",
		"今後は計算量固定のFID-50kを完了させ、両条件をFID-50kで比較します。",
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
	console.log("fixed main18 FID-50k:", main18Fid50k.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log("fixed NFE55 FID-25k:", nfe55Fid25k.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log(nfe55Fid50kText);
})();
