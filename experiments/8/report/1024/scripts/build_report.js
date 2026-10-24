// Build the 1024 progress report deck: the two Restart K sweeps (t_min = 0.14), the 3-seed
// check of the optimal K, and training past 200k.
//
// Usage (from the EDM repository root, after scripts/prepare_figures.py):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/1024/scripts/build_report.js
//
// Numbers come from the snapshot in assets/. Sentences that describe them are written by hand,
// so the build stops when the snapshot no longer matches them.

const fs = require("fs");
const path = require("path");
const assert = require("assert");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report", "1024");
const ASSETS = path.join(REPORT_DIR, "assets");
const OUT = path.join(REPORT_DIR, "restart_progress_report_1024_claude.pptx");

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

function readTsv(file) {
	const [header, ...rows] = fs.readFileSync(path.join(ASSETS, file), "utf8").trim().split("\n").map((l) => l.split("\t"));
	return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

// fid_by_checkpoint_and_k.tsv -> { stage: { ckpt: { k: fid } } }
function readSweep(file) {
	const out = {};
	for (const r of readTsv(file)) {
		((out[r.num_images] = out[r.num_images] || {})[r.checkpoint] = out[r.num_images][r.checkpoint] || {})[Number(r.restart_k)] = Number(r.fid);
	}
	return out;
}

const byKimg = (a, b) => parseInt(a, 10) - parseInt(b, 10);

// Best K and gain over K=0 per checkpoint, for checkpoints with all nine K evaluated.
function bestK(stageTable) {
	return Object.keys(stageTable).sort(byKimg)
		.filter((ckpt) => Object.keys(stageTable[ckpt]).length === 9)
		.map((ckpt) => {
			const fids = stageTable[ckpt];
			const k = Object.keys(fids).map(Number).reduce((a, b) => (fids[b] < fids[a] ? b : a));
			return { ckpt, k, gain: fids[0] - fids[k] };
		});
}

const MAIN18 = bestK(readSweep("fixed_main18__fid_by_checkpoint_and_k.tsv")["50000"]);
const NFE55 = bestK(readSweep("fixed_nfe55__fid_by_checkpoint_and_k.tsv")["50000"]);
const PROGRESS = JSON.parse(fs.readFileSync(path.join(ASSETS, "progress.json"), "utf8"));
const TMIN = JSON.parse(fs.readFileSync(path.join(ASSETS, "actual_tmin.json"), "utf8"));
const SEEDS3_COND = readTsv("seeds3__condition_stats.tsv");
const SEEDS3_GAIN = readTsv("seeds3__improvement_stats.tsv");
const POST200K = readTsv("post200k__comparison.tsv");

// The hand-written sentences below describe exactly these results.
assert.deepStrictEqual(MAIN18.map((r) => r.k), [7, 3, 3, 3, 3, 3, 1, 3, 1, 1]);
assert.deepStrictEqual(NFE55.map((r) => r.k), [8, 6, 4, 6, 4, 7, 7, 1, 1, 1]);
assert.ok(NFE55.every((r) => r.gain > 0));
for (const sweep of ["fixed_main18", "fixed_nfe55"]) for (const stage of ["5000", "25000", "50000"]) assert.strictEqual(PROGRESS[sweep][stage], 90);
assert.ok(SEEDS3_GAIN.every((r) => Number(r.n) === 0 || Number(r.positive_ranges) === Number(r.n)));
assert.ok(POST200K.every((r) => Number(r.improvement_mean) < 0));
assert.strictEqual(PROGRESS.post200k_last_snapshot_kimg, 220000);

const fmtGain = (g) => `${g >= 0 ? "+" : "−"}${Math.abs(g).toFixed(3)}`;
const seeds3Complete = SEEDS3_GAIN.filter((r) => r.complete === "True").map((r) => r.checkpoint);
const seeds3Total = SEEDS3_COND.length * 3;

// ---------------------------------------------------------------------------- deck

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "研究進捗";
const C = pres.SchemeColor;
const ORANGE = C.accent2, GREEN = C.accent4, NAVY = C.text2, TINT = C.background2, RED = C.accent6;

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

// A chip and one line of text, as the condition / status lines under a title.
function infoLine(slide, label, value, y, fill) {
	chip(slide, label, MX, y, 0.8, fill);
	text(slide, value, { x: MX + 0.95, y: y + 0.02, w: CW - 0.95, h: 0.3, fontSize: 13 });
}

function picture(slide, file, x, y, w, aspect, name) {
	slide.addImage({ path: path.join(ASSETS, file), x, y, w, h: w / aspect, objectName: name });
	return w / aspect;
}

// Fresh options per cell: pptxgenjs mutates option objects in place.
const head = (fontSize) => ({ bold: true, color: C.background1, fill: { color: NAVY }, align: "center", valign: "middle", fontSize });
const rowLabel = () => ({ bold: true, color: NAVY, align: "left" });
const gainCell = (g) => ({ text: fmtGain(g), options: { bold: true, color: g > 0 ? GREEN : RED } });
const BORDER = () => ({ type: "solid", pt: 1, color: "D9DCDF" });

// ---------------------------------------------------------------------------- 1. overview

pres.addSection({ title: "概要" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "概要" });
	s.addText("研究進捗", { placeholder: "title" });
	text(s, "CIFAR-10 class-conditional EDM、10k間隔で再学習したcheckpoint系列、Restartの開始ノイズ量 t_min = 0.14", { x: MX, y: 1.02, w: CW, h: 0.35, fontSize: 16, color: C.accent5 });

	const status = (label, color, detail) => ({ text: [
		{ text: label, options: { bold: true, color, breakLine: true } },
		{ text: detail, options: { fontSize: 13 } },
	] });
	const name = (value) => ({ text: value, options: { bold: true, color: NAVY } });
	const rows = [
		[{ text: "実験", options: head(14) }, { text: "目的", options: head(14) }, { text: "進捗", options: head(14) }],
		[name("主経路を固定したK探索"), "学習段階ごとの最良Kを調べる（Kを増やすと計算量も増える）",
			status("完了", GREEN, "FID-5k・25k・50kとも全90条件（20k〜200k）")],
		[name("計算量を固定したK探索"), "同じ55 NFEで、主経路とRestartへの計算の配分を比べる",
			status("完了", GREEN, "FID-5k・25k・50kとも全90条件（20k〜200k）")],
		[name("最適KとK=0の3シード比較"), "最適Kによる改善が、生成シードを変えても再現するか確かめる",
			status("評価中", ORANGE, `${PROGRESS.seeds3_done} / ${seeds3Total}評価。${seeds3Complete.join("・")}は3範囲とも完了`)],
		[name("200k以降の継続学習"), "200kから学習を続けて、FIDが改善するか確かめる",
			status("220kで停止", RED, "FID-50kは210k・220kとも200kより悪化（1シード範囲）")],
	];
	s.addTable(rows, {
		x: MX, y: 1.65, w: CW, colW: [3.4, 4.3, CW - 7.7], rowH: [0.45, 0.95, 0.95, 0.95, 0.95],
		fontSize: 14, color: C.text1, valign: "middle", margin: [0.06, 0.15, 0.06, 0.15], border: BORDER(),
	});

	s.addNotes([
		"今回は4つの実験の進捗を報告します。どれも10k間隔で再学習したcheckpoint系列を使い、Restartの開始ノイズ量t_minは0.14です。",
		"前回途中だった計算量を固定したK探索は、20kから200kまでFID-50kが完了しました。主経路を固定したK探索と合わせて、両条件をFID-50kで比較できます。",
		`新しく、主経路固定で選んだ最適KがK=0より本当に良いかを、生成シードを3通りに変えて確認しています。現在${PROGRESS.seeds3_done}/${seeds3Total}評価まで終わっています。もう1つは200kからの継続学習で、220kまで学習しましたが、FIDは悪化しました。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 2. K sweep conditions

pres.addSection({ title: "K探索" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "K探索" });
	s.addText("K探索の2つの条件", { placeholder: "title" });
	text(s, "開始ノイズ量 t_min = 0.14、t_max = 0.30、K = 0〜8、各条件50,000枚（FID-50k）", { x: MX, y: 1.02, w: CW, h: 0.35, fontSize: 16, color: C.accent5 });

	const rows = [
		[{ text: "", options: head(14) }, { text: "主経路を固定", options: head(15) }, { text: "計算量を固定", options: head(15) }],
		[{ text: "考え方", options: rowLabel() }, "主経路はそのまま。Restartを足すほど計算量が増える", "計算量をそろえ、主経路を減らした分をRestartに回す"],
		[{ text: "主経路", options: rowLabel() }, "18 steps（全K共通）", "28−2K steps（K=0：28 → K=8：12）"],
		[{ text: "NFE", options: rowLabel() }, "35+4K（35〜67）", "55（全K共通）"],
		[{ text: "K=0（Restartなし）", options: rowLabel() }, "通常ODE 18 steps（35 NFE）", "通常ODE 28 steps（55 NFE）"],
		[{ text: "checkpoint", options: rowLabel() }, `20k〜200k（${MAIN18.length}点）`, `20k〜200k（${NFE55.length}点）`],
	];
	s.addTable(rows, {
		x: MX, y: 1.65, w: CW, colW: [2.5, (CW - 2.5) / 2, (CW - 2.5) / 2], rowH: [0.45, 0.7, 0.6, 0.6, 0.6, 0.6],
		fontSize: 14, color: C.text1, valign: "middle", margin: [0.05, 0.15, 0.05, 0.15], border: BORDER(),
	});

	s.addNotes([
		"K探索は2つの条件で行いました。主経路を固定した方は、主経路を18 stepsのままRestartを足すので、Kを増やすほど計算量も増えます。",
		"計算量を固定した方は、NFEを55にそろえ、Kを1増やすごとに主経路を2 steps減らします。こちらのK=0は28 stepsの通常ODEなので、同じ計算量のODEとの比較になります。",
		"今回はどちらも20kから200kまでの10点で、FID-50kまで揃いました。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 3-4. K sweep results

const FIG_ASPECT = 2880 / 1650;

function sweepSlide(title, condition, figure, results, gainNote, notes) {
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "K探索" });
	s.addText(title, { placeholder: "title" });
	infoLine(s, "条件", condition, 1.13);

	const fh = 5.2;
	picture(s, figure, MX, 1.6, fh * FIG_ASPECT, FIG_ASPECT, `figure ${title}`);
	const tx = MX + fh * FIG_ASPECT + 0.2, tw = W - MX - tx;
	const rows = [[{ text: "checkpoint", options: head(11) }, { text: "最良K", options: head(11) }, { text: "改善幅", options: head(11) }]]
		.concat(results.map((r) => [r.ckpt, `K=${r.k}`, gainCell(r.gain)]));
	s.addTable(rows, {
		x: tx, y: 1.6, w: tw, colW: [tw * 0.36, tw * 0.28, tw * 0.36], rowH: 0.29,
		fontSize: 11.5, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.04, 0.01, 0.04], border: BORDER(),
	});
	text(s, [
		{ text: `改善幅 = FID(K=0) − 最小FID。${gainNote}`, options: { breakLine: true } },
		{ text: "左上：KごとのFID（低いほど良い）", options: { breakLine: true } },
		{ text: "右上：FID(K) − FID(K=0)（青ほど改善、赤ほど悪化）", options: { breakLine: true } },
		{ text: "左下：各checkpointの最良K", options: { breakLine: true } },
		{ text: "右下：K=0のFIDと改善幅の関係" },
	], { x: tx, y: 1.6 + 0.29 * 11 + 0.15, w: tw, h: 1.9, fontSize: 10.5, color: MUTED, paraSpaceAfter: 3 });
	s.addNotes(notes.join("\n\n"));
}

sweepSlide("主経路を固定したK探索（FID-50k）",
	"主経路18 steps、K=0〜8、NFE = 35+4K（35〜67）、t_min = 0.14、各条件50,000枚、20k〜200k",
	"fixed_main18__fid50000.png", MAIN18, "Kとともに計算量も増える。",
	[
		"主経路を18 stepsに固定した探索のFID-50kです。前回と同じ結果で、図だけ4枚構成に描き直しています。",
		`最良Kは、20kでK=7、40kから120kではK=3、140k以降はK=1〜3です。K=0に対する改善幅は20kの${fmtGain(MAIN18[0].gain)}から200kの${fmtGain(MAIN18[MAIN18.length - 1].gain)}まで小さくなりました。`,
		"右下の図のとおり、K=0のFIDが大きい、つまり学習初期のモデルほどRestartによる改善が大きくなっています。",
	]);

sweepSlide("計算量を固定したK探索（FID-50k）",
	"NFE = 55で固定、主経路28−2K steps、K=0〜8、t_min = 0.14、各条件50,000枚、20k〜200k",
	"fixed_nfe55__fid50000.png", NFE55, "K=0は同じ55 NFEの通常ODE（28 steps）。",
	[
		"計算量を55 NFEに固定した探索のFID-50kです。前回はFID-25kの20k〜120kまででしたが、今回20k〜200kのFID-50kが揃いました。",
		`ここでのK=0は28 stepsの通常ODEなので、改善幅は同じ計算量のODEに対する改善です。全checkpointでRestartの方が良く、改善幅は20kの${fmtGain(NFE55[0].gain)}から200kの${fmtGain(NFE55[NFE55.length - 1].gain)}まで小さくなりました。`,
		"最良Kは、20kでK=8、40k〜140kではK=4〜7とばらつき、160k以降はK=1でした。次のページで説明するように、この条件ではKによって実際の開始ノイズ量も変わるので、その影響も含まれています。",
	]);

// ---------------------------------------------------------------------------- 5. actual t_min per K

{
	const main18 = TMIN.fixed_main18, nfe55 = TMIN.fixed_nfe55;
	const ks = [...Array(9).keys()];
	const even = (k) => k > 0 && k % 2 === 0;
	assert.ok(main18.slice(1).every((r) => r.actual_t_min === main18[1].actual_t_min));
	assert.ok(Math.max(...ks.filter(even).map((k) => nfe55[k].actual_t_min)) < Math.min(...ks.filter((k) => k % 2 === 1).map((k) => nfe55[k].actual_t_min)));

	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "K探索" });
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
		fontSize: 13, color: C.text1, align: "center", valign: "middle", margin: [0.02, 0.06, 0.02, 0.06], border: BORDER(),
	});
	text(s, "K=0はRestartなし（—）。色付きのセル：計算量固定で実際の開始ノイズ量が小さい偶数K。", { x: MX, y: 5.02, w: CW, h: 0.28, fontSize: 11, color: MUTED });

	s.addNotes([
		`Restartはt_min=0.14を指定しても、主経路のノイズスケジュール上で0.14に最も近い点から始まります。主経路固定ではK=1〜8のすべてで${main18[1].actual_t_min.toFixed(4)}です。`,
		"計算量固定では主経路が28−2K stepsと変わるため、実際の開始ノイズ量もKごとに変わります。奇数Kでは0.13〜0.15ですが、偶数Kでは0.065〜0.109と小さくなります。計算量固定の最良Kには、Kの効果と開始ノイズ量の効果が混ざっている可能性があります。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 6. 3-seed check

pres.addSection({ title: "追加実験" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "追加実験" });
	s.addText("最適KとK=0の比較（FID-50k・3シード範囲）", { placeholder: "title" });
	infoLine(s, "条件", "主経路固定のK探索で選んだ最適Kを固定。生成seedの3範囲（0–49,999／50,000–99,999／100,000–149,999）で評価", 1.13);
	const pending = SEEDS3_GAIN.filter((r) => r.complete !== "True").map((r) => r.checkpoint);
	if (pending.length) {
		infoLine(s, "評価中", `${PROGRESS.seeds3_done} / ${seeds3Total}評価。${pending.join("・")}は1範囲のみ（残りを評価中）`, 1.53, ORANGE);
	}

	const fw = 8.7;
	const fh = picture(s, "seeds3__fid50k_3seeds.png", (W - fw) / 2, 1.95, fw, 1200 / 400, "figure 3-seed comparison");
	text(s, "左：FID-50k（灰：K=0、青：最適K。各点が1シード範囲）。右：改善幅 = FID(K=0) − FID(最適K)（黒：平均 ± 標本SD）。",
		{ x: MX, y: 1.95 + fh + 0.03, w: CW, h: 0.28, fontSize: 10.5, color: MUTED });

	const cond = Object.fromEntries(SEEDS3_COND.map((r) => [r.condition, r]));
	const meanSd = (r) => (Number(r.n) > 1 ? `${Number(r.mean).toFixed(3)} ± ${Number(r.std).toFixed(3)}` : `${Number(r.mean).toFixed(3)}`);
	const partial = (r, value) => (r.complete === "True" ? value : { text: `${typeof value === "string" ? value : value.text}*`, options: { color: ORANGE } });
	const ckpts = SEEDS3_GAIN;
	const rows = [
		[{ text: "checkpoint", options: head(11) }].concat(ckpts.map((r) => ({ text: r.checkpoint, options: head(11) }))),
		[{ text: "最適K（NFE）", options: rowLabel() }].concat(ckpts.map((r) => `K=${r.optimal_k}（${35 + 4 * Number(r.optimal_k)}）`)),
		[{ text: "K=0のFID", options: rowLabel() }].concat(ckpts.map((r) => partial(r, meanSd(cond[`${r.checkpoint}-k0`])))),
		[{ text: "最適KのFID", options: rowLabel() }].concat(ckpts.map((r) => partial(r, meanSd(cond[`${r.checkpoint}-k${r.optimal_k}`])))),
		[{ text: "改善幅", options: rowLabel() }].concat(ckpts.map((r) => (r.complete === "True"
			? { text: `${fmtGain(Number(r.mean))} ± ${Number(r.std).toFixed(3)}`, options: { bold: true, color: GREEN } }
			: { text: `${fmtGain(Number(r.mean))}*`, options: { bold: true, color: ORANGE } }))),
		[{ text: "改善した範囲", options: rowLabel() }].concat(ckpts.map((r) => partial(r, `${r.positive_ranges} / ${r.n}`))),
	];
	const firstW = 1.9;
	s.addTable(rows, {
		x: MX, y: 5.3, w: CW, colW: [firstW].concat(ckpts.map(() => (CW - firstW) / ckpts.length)), rowH: 0.24,
		fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.04, 0.01, 0.04], border: BORDER(),
	});
	if (pending.length) text(s, "*評価途中（1シード範囲のみ）", { x: MX, y: 6.78, w: 4, h: 0.22, fontSize: 9.5, color: ORANGE });

	const done = SEEDS3_GAIN.filter((r) => r.complete === "True");
	s.addNotes([
		"主経路を固定したK探索で選んだ最適Kが、たまたま良かったのではないかを確認するため、同じモデルで生成シードの範囲を3通りに変えてFID-50kを評価しています。学習は1回だけで、変えているのは生成のシードです。",
		`3範囲が揃った${done.map((r) => r.checkpoint).join("・")}では、どの範囲でも最適KがK=0より良くなりました。改善幅の平均は${done.map((r) => `${r.checkpoint}で${fmtGain(Number(r.mean))}`).join("、")}です。`,
		pending.length ? `${pending.join("・")}はまだ1範囲だけで、残りを評価中です。` : "全checkpointで3範囲の評価が完了しています。",
		"後期のcheckpointでは改善幅が0.01〜0.02と小さく、シードによる差と同じ程度なので、効果はほとんど残っていません。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 7. training past 200k

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "追加実験" });
	s.addText("200k以降の継続学習", { placeholder: "title" });
	infoLine(s, "条件", "200kから学習率0.001のまま継続学習（EMA 0.5 Mimg、学習seed 0、2.5kごとに保存）。220kで停止", 1.13);
	infoLine(s, "評価", "FID-50k（seed 0–49,999の1範囲）。K=0（通常ODE 18 steps、35 NFE）とK=1（200kの最適K、39 NFE）", 1.53);

	const get = (kimg, k) => POST200K.find((r) => r.kimg === String(kimg) && r.restart_k === String(k));
	const series = [0, 1].map((k) => ({
		name: k === 0 ? "K=0（35 NFE）" : "K=1（39 NFE）",
		labels: ["200k", "210k", "220k"],
		values: [Number(get(210000, k).baseline_mean), Number(get(210000, k).candidate_mean), Number(get(220000, k).candidate_mean)],
	}));
	s.addChart(pres.charts.LINE, series, {
		x: MX, y: 2.05, w: 7.4, h: 4.75,
		chartColors: [THEME.colors.accent5, THEME.colors.accent2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 9,
		showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
		valAxisMinVal: 1.75, valAxisMaxVal: 2.1, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
		valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
		valAxisLabelColor: MUTED, catAxisLabelColor: MUTED,
		showValAxisTitle: true, valAxisTitle: "FID-50k（低いほど良い）", valAxisTitleFontSize: 11, valAxisTitleColor: MUTED, valAxisTitleFontFace: "+mn-lt",
		showCatAxisTitle: true, catAxisTitle: "学習量（kimg）", catAxisTitleFontSize: 11, catAxisTitleColor: MUTED, catAxisTitleFontFace: "+mn-lt",
		valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
	});

	const tx = MX + 7.4 + 0.35, tw = W - MX - tx;
	// FID change from 200k; positive means worse.
	const cell = (value, base) => (base === undefined ? value.toFixed(3) : { text: [
		{ text: value.toFixed(3), options: { breakLine: true } },
		{ text: `（200k比 ${fmtGain(value - base)}）`, options: { fontSize: 10, color: value > base ? RED : GREEN } },
	] });
	const rows = [[{ text: "学習量", options: head(12) }, { text: "K=0", options: head(12) }, { text: "K=1", options: head(12) }]];
	["200k", "210k", "220k"].forEach((label, i) => {
		rows.push([{ text: label, options: rowLabel() }].concat(series.map((sr) => cell(sr.values[i], i === 0 ? undefined : sr.values[0]))));
	});
	s.addTable(rows, {
		x: tx, y: 2.1, w: tw, colW: [1.25, (tw - 1.25) / 2, (tw - 1.25) / 2], rowH: [0.4, 0.5, 0.7, 0.7],
		fontSize: 13, color: C.text1, align: "center", valign: "middle", margin: [0.03, 0.05, 0.03, 0.05], border: BORDER(),
	});
	card(s, tx, 4.75, tw, 1.55, TINT, "result");
	chip(s, "結果", tx + 0.2, 4.92, 0.8, RED);
	text(s, "210k・220kとも、K=0・K=1の両方で200kよりFIDが悪化した。1シード範囲のスクリーニング結果", { x: tx + 0.2, y: 5.32, w: tw - 0.4, h: 0.9, fontSize: 13 });

	const k0 = series[0].values, k1 = series[1].values;
	s.addNotes([
		"最後に、200kからさらに学習を続けた場合です。学習率0.001のまま継続し、2.5kごとに保存しながら220kまで学習しました。",
		`FID-50kはK=0で${k0.map((v) => v.toFixed(3)).join(" → ")}、K=1で${k1.map((v) => v.toFixed(3)).join(" → ")}と、210k・220kのどちらも200kより悪化しました。特に220kでは大きく悪化しています。`,
		"ただし、これは生成シード1範囲だけのスクリーニングです。学習率を下げるなどの条件で継続するかは、今後検討します。",
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
	console.log("fixed main18 FID-50k:", MAIN18.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log("fixed NFE55 FID-50k:", NFE55.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log(`3-seed: ${PROGRESS.seeds3_done}/${seeds3Total}, complete: ${seeds3Complete.join(",")}`);
})();
