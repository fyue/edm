// Build the 1107 progress report deck: the two Restart K sweeps (t_min = 0.14), the 3-seed
// check of the optimal K, training past 200k, and the 30,000-image training with its K sweeps.
//
// Usage (from the EDM repository root, after scripts/prepare_figures.py):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/1107/scripts/build_report.js
//
// Numbers come from the snapshot in assets/. Sentences that describe them are written by hand,
// so the build stops when the snapshot no longer matches them.

const fs = require("fs");
const path = require("path");
const assert = require("assert");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report", "1107");
const ASSETS = path.join(REPORT_DIR, "assets");
const OUT = path.join(REPORT_DIR, "restart_progress_report_1107_claude.pptx");

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

// Mean of the K within 0.02 of the minimum FID, as drawn in the near-optimal K panel.
function meanK(fids) {
	const best = Math.min(...Object.values(fids));
	const near = Object.keys(fids).map(Number).filter((k) => fids[k] - best <= 0.02);
	return near.reduce((a, b) => a + b, 0) / near.length;
}

const MAIN18_FID = readSweep("fixed_main18__fid_by_checkpoint_and_k.tsv")["50000"];
const NFE55_FID = readSweep("fixed_nfe55__fid_by_checkpoint_and_k.tsv")["50000"];
// The 30,000-image K sweeps are reported at these checkpoints only.
const SUBSET_CKPTS = ["20k", "40k", "60k"];
const subsetSweep = (file) => {
	const table = readSweep(file)["50000"];
	return Object.fromEntries(SUBSET_CKPTS.map((ckpt) => [ckpt, table[ckpt]]));
};
const SUBSET_MAIN18_FID = subsetSweep("subset_main18__fid_by_checkpoint_and_k.tsv");
const SUBSET_NFE55_FID = subsetSweep("subset_nfe55__fid_by_checkpoint_and_k.tsv");
const MAIN18 = bestK(MAIN18_FID);
const NFE55 = bestK(NFE55_FID);
const SUBSET_MAIN18 = bestK(SUBSET_MAIN18_FID);
const SUBSET_NFE55 = bestK(SUBSET_NFE55_FID);
const PROGRESS = JSON.parse(fs.readFileSync(path.join(ASSETS, "progress.json"), "utf8"));
const TMIN = JSON.parse(fs.readFileSync(path.join(ASSETS, "actual_tmin.json"), "utf8"));
const SEEDS3_COND = readTsv("seeds3__condition_stats.tsv");
const SEEDS3_GAIN = readTsv("seeds3__improvement_stats.tsv");
const POST200K = readTsv("post200k__comparison.tsv");
const POST200K_COND = readTsv("post200k_seeds3__condition_stats.tsv");
const POST200K_GAIN = readTsv("post200k_seeds3__improvement_stats.tsv");
const SUBSET_TRAIN = readTsv("subset__fid_summary.tsv").map((r) => ({ kimg: Number(r.kimg), fid: Number(r.fid) }));

// The hand-written sentences below describe exactly these results.
assert.deepStrictEqual(MAIN18.map((r) => r.k), [7, 3, 3, 3, 3, 3, 1, 3, 1, 1]);
assert.deepStrictEqual(NFE55.map((r) => r.k), [8, 6, 4, 6, 4, 7, 7, 1, 1, 1]);
assert.ok(NFE55.every((r) => r.gain > 0));
for (const sweep of ["fixed_main18", "fixed_nfe55"]) for (const stage of ["5000", "25000", "50000"]) assert.strictEqual(PROGRESS[sweep][stage], 90);
assert.ok(SEEDS3_GAIN.every((r) => r.complete === "True" && Number(r.n) === 3 && Number(r.positive_ranges) === 3));
assert.ok(POST200K_GAIN.every((r) => r.complete === "True" && Number(r.n) === 3 && Number(r.positive_ranges) === 3));
assert.deepStrictEqual(SUBSET_TRAIN.map((r) => r.kimg), [...Array(10).keys()].map((i) => (i + 1) * 10000));
assert.strictEqual(SUBSET_TRAIN.reduce((a, b) => (b.fid < a.fid ? b : a)).kimg, 50000);
assert.deepStrictEqual(SUBSET_MAIN18.map((r) => [r.ckpt, r.k]), [["20k", 3], ["40k", 0], ["60k", 0]]);
assert.deepStrictEqual(SUBSET_NFE55.map((r) => [r.ckpt, r.k]), [["20k", 6], ["40k", 0], ["60k", 0]]);
for (const table of [SUBSET_MAIN18_FID, SUBSET_NFE55_FID]) {
	for (const ckpt of ["40k", "60k"]) {
		assert.ok([1, 2, 3, 4, 5, 6, 7, 8].every((k) => table[ckpt][k] > table[ckpt][0]));
	}
}
assert.ok(POST200K.every((r) => Number(r.improvement_mean) < 0));
assert.strictEqual(PROGRESS.post200k_last_snapshot_kimg, 220000);

const fmtGain = (g) => `${g >= 0 ? "+" : "−"}${Math.abs(g).toFixed(3)}`;
const meanSd = (mean, std, digits = 3) => `${Number(mean).toFixed(digits)} ± ${Number(std).toFixed(digits)}`;
const seeds3Total = SEEDS3_COND.length * 3;
const seeds3Done = SEEDS3_COND.reduce((sum, r) => sum + Number(r.n), 0);
assert.strictEqual(seeds3Done, seeds3Total);
const subsetBest = SUBSET_TRAIN.reduce((a, b) => (b.fid < a.fid ? b : a));
const subsetLast = SUBSET_TRAIN[SUBSET_TRAIN.length - 1];

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
	text(s, "CIFAR-10 class-conditional EDM、Restartの開始ノイズ量 t_min = 0.14", { x: MX, y: 1.02, w: CW, h: 0.35, fontSize: 16, color: C.accent5 });

	const status = (label, color, detail) => ({ text: [
		{ text: label, options: { bold: true, color, breakLine: true } },
		{ text: detail, options: { fontSize: 12 } },
	] });
	const name = (value) => ({ text: value, options: { bold: true, color: NAVY } });
	const rows = [
		[{ text: "実験", options: head(14) }, { text: "目的", options: head(14) }, { text: "進捗", options: head(14) }],
		[name("主経路を固定したK探索"), "学習段階ごとの最良Kを調べる（Kを増やすと計算量も増える）",
			status("完了", GREEN, "FID-5k・25k・50kとも全90条件（20k〜200k）")],
		[name("計算量を固定したK探索"), "同じ55 NFEで、主経路とRestartへの計算の配分を比べる",
			status("完了", GREEN, "FID-5k・25k・50kとも全90条件（20k〜200k）")],
		[name("最適KとK=0の3シード比較"), "最適Kによる改善が、生成シードを変えても再現するか確かめる",
			status("完了", GREEN, `${seeds3Done} / ${seeds3Total}評価。6点すべて、3範囲とも改善`)],
		[name("200k以降の継続学習"), "200kから学習を続けて、FIDが改善するか確かめる",
			status("220kで停止", RED, "FIDは210k・220kとも200kより悪化。200k・210kのK=1は3範囲で確認")],
		[name("30,000枚の学習データで再学習"), "データを減らしたときのFIDの推移を確かめる",
			status("完了", GREEN, `100kまで学習。FID-50kは${subsetBest.kimg / 1000}kの${subsetBest.fid.toFixed(3)}が最良で、以降は悪化`)],
		[name("30,000枚モデルのK探索"), "データが少ないモデルでもRestartで改善するか確かめる",
			status("20k・40k・60kで評価", GREEN, "FID-50kを2条件で比較。Restartで改善したのは20kのみ")],
	];
	s.addTable(rows, {
		x: MX, y: 1.5, w: CW, colW: [3.5, 4.2, CW - 7.7], rowH: [0.42, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78],
		fontSize: 13, color: C.text1, valign: "middle", margin: [0.04, 0.15, 0.04, 0.15], border: BORDER(),
	});

	s.addNotes([
		"今回は6つの実験の進捗を報告します。Restartの開始ノイズ量t_minは、どれも0.14です。",
		`K探索の2条件は前回完了済みで、今回は図を更新しました。前回途中だった3シード比較は${seeds3Done}評価すべて完了し、200k以降の継続学習も200k・210kを3範囲で確認しました。`,
		"新しく、学習データを30,000枚に減らした再学習と、そのモデルでのK探索を報告します。再学習は100kまで完了し、K探索は20k・40k・60kの結果を示します。",
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
		"どちらも20kから200kまでの10点で、FID-50kまで揃っています。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 3-4. K sweep results

const FIG_ASPECT = 2880 / 1650;
const MEAN_K_SERIES = "平均K";
const BEST_K_SERIES = "最良K";

// A figure on the left, and a per-checkpoint table with panel notes on the right.
function sweepSlide({ title, condition, figure, rows, captions, notes }) {
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "K探索" });
	s.addText(title, { placeholder: "title" });
	infoLine(s, "条件", condition, 1.13);

	const top = 1.6, fh = 6.85 - top;
	picture(s, figure, MX, top, fh * FIG_ASPECT, FIG_ASPECT, `figure ${title}`);
	const tx = MX + fh * FIG_ASPECT + 0.2, tw = W - MX - tx;
	s.addTable([["checkpoint", "最良K", "改善幅"].map((c) => ({ text: c, options: head(11) }))].concat(rows), {
		x: tx, y: top, w: tw, colW: [0.42, 0.24, 0.34].map((w) => w * tw), rowH: 0.29,
		fontSize: 11.5, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.04, 0.01, 0.04], border: BORDER(),
	});
	text(s, captions.map((c, i) => ({ text: c, options: { breakLine: i < captions.length - 1 } })),
		{ x: tx, y: top + 0.29 * (rows.length + 1) + 0.15, w: tw, h: 6.85 - (top + 0.29 * (rows.length + 1) + 0.15), fontSize: 10.5, color: MUTED, paraSpaceAfter: 3 });
	s.addNotes(notes.join("\n\n"));
}

const FULL_CAPTIONS = [
	"左上：KごとのFID（低いほど良い）",
	"右上：FID(K) − FID(K=0)（青ほど改善、赤ほど悪化）",
	"左下：黒は最良K、青は最良FIDとの差が0.02以下のK、橙はその単純平均",
	"右下：K=0のFIDと改善幅の関係",
];
const bestKRows = (results) => results.map((r) => [r.ckpt, `K=${r.k}`, gainCell(r.gain)]);
const meanKOf = (table, ckpt) => meanK(table[ckpt]).toFixed(1);

sweepSlide({
	title: "主経路を固定したK探索（FID-50k）",
	condition: "主経路18 steps、K=0〜8、NFE = 35+4K（35〜67）、t_min = 0.14、各条件50,000枚、20k〜200k",
	figure: "fixed_main18__fid50000_mean_k.png",
	rows: bestKRows(MAIN18),
	captions: ["改善幅 = FID(K=0) − 最小FID。Kとともに計算量も増える。"].concat(FULL_CAPTIONS),
	notes: [
		"主経路を18 stepsに固定した探索のFID-50kです。数値は前回と同じで、左下の図を差し替えました。",
		`左下の黒は最良K、青は最良FIDとの差が0.02以下のK、橙はその単純平均です。最良Kだけを見るとばらつきますが、平均は20kの${meanKOf(MAIN18_FID, "20k")}から200kの${meanKOf(MAIN18_FID, "200k")}まで下がっていて、学習が進むほど小さいKで十分という傾向が見えます。`,
		`K=0に対する改善幅は20kの${fmtGain(MAIN18[0].gain)}から200kの${fmtGain(MAIN18[MAIN18.length - 1].gain)}まで小さくなりました。右下の図のとおり、K=0のFIDが大きい、つまり学習初期のモデルほどRestartによる改善が大きくなっています。`,
	],
});

sweepSlide({
	title: "計算量を固定したK探索（FID-50k）",
	condition: "NFE = 55で固定、主経路28−2K steps、K=0〜8、t_min = 0.14、各条件50,000枚、20k〜200k",
	figure: "fixed_nfe55__fid50000_mean_k.png",
	rows: bestKRows(NFE55),
	captions: ["改善幅 = FID(K=0) − 最小FID。K=0は同じ55 NFEの通常ODE（28 steps）。"].concat(FULL_CAPTIONS),
	notes: [
		"計算量を55 NFEに固定した探索のFID-50kです。こちらも数値は前回と同じで、左下の図を差し替えました。",
		`ここでのK=0は28 stepsの通常ODEなので、改善幅は同じ計算量のODEに対する改善です。全checkpointでRestartの方が良く、改善幅は20kの${fmtGain(NFE55[0].gain)}から200kの${fmtGain(NFE55[NFE55.length - 1].gain)}まで小さくなりました。`,
		`近い性能のKの平均は、20kの${meanKOf(NFE55_FID, "20k")}から200kの${meanKOf(NFE55_FID, "200k")}まで下がります。ただし途中はばらつきがあります。次のページで説明するように、この条件ではKによって実際の開始ノイズ量も変わるので、その影響も含まれています。`,
	],
});

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

	const fw = 9.9;
	const fh = picture(s, "seeds3__fid50k_3seeds.png", (W - fw) / 2, 1.55, fw, 1200 / 400, "figure 3-seed comparison");
	text(s, "左：FID-50k（灰：K=0、青：最適K。各点が1シード範囲）。右：改善幅 = FID(K=0) − FID(最適K)（黒：平均 ± 標本SD）。",
		{ x: MX, y: 1.55 + fh + 0.03, w: CW, h: 0.28, fontSize: 10.5, color: MUTED });

	const cond = Object.fromEntries(SEEDS3_COND.map((r) => [r.condition, r]));
	const fidCell = (r) => meanSd(r.mean, r.std);
	const ckpts = SEEDS3_GAIN;
	const rows = [
		[{ text: "checkpoint", options: head(11) }].concat(ckpts.map((r) => ({ text: r.checkpoint, options: head(11) }))),
		[{ text: "最適K（NFE）", options: rowLabel() }].concat(ckpts.map((r) => `K=${r.optimal_k}（${35 + 4 * Number(r.optimal_k)}）`)),
		[{ text: "K=0のFID", options: rowLabel() }].concat(ckpts.map((r) => fidCell(cond[`${r.checkpoint}-k0`]))),
		[{ text: "最適KのFID", options: rowLabel() }].concat(ckpts.map((r) => fidCell(cond[`${r.checkpoint}-k${r.optimal_k}`]))),
		[{ text: "改善幅", options: rowLabel() }].concat(ckpts.map((r) => ({ text: `${fmtGain(Number(r.mean))} ± ${Number(r.std).toFixed(3)}`, options: { bold: true, color: GREEN } }))),
		[{ text: "改善した範囲", options: rowLabel() }].concat(ckpts.map((r) => `${r.positive_ranges} / ${r.n}`)),
	];
	const firstW = 1.9;
	s.addTable(rows, {
		x: MX, y: 5.22, w: CW, colW: [firstW].concat(ckpts.map(() => (CW - firstW) / ckpts.length)), rowH: 0.25,
		fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.04, 0.01, 0.04], border: BORDER(),
	});

	s.addNotes([
		"主経路を固定したK探索で選んだ最適Kが、たまたま良かったのではないかを確認するため、同じモデルで生成シードの範囲を3通りに変えてFID-50kを評価しました。学習は1回だけで、変えているのは生成のシードです。",
		`前回途中だった100kと140kも揃い、6点すべてで、どの範囲でも最適KがK=0より良くなりました。改善幅の平均は${ckpts.map((r) => `${r.checkpoint}で${fmtGain(Number(r.mean))}`).join("、")}です。`,
		"後期のcheckpointでは改善幅が0.01〜0.02と小さく、標準偏差と同じ程度です。符号は3範囲とも改善ですが、効果そのものはかなり小さくなっています。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 7. training past 200k

{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "追加実験" });
	s.addText("200k以降の継続学習", { placeholder: "title" });
	infoLine(s, "条件", "200kから学習率0.001のまま継続学習（EMA 0.5 Mimg、学習seed 0、2.5kごとに保存）。220kで停止", 1.13);
	infoLine(s, "評価", "FID-50k。K=0（通常ODE、35 NFE）とK=1（39 NFE）。200k・210kは3シード範囲、220kは1範囲", 1.53);

	const get = (kimg, k) => POST200K.find((r) => r.kimg === String(kimg) && r.restart_k === String(k));
	const series = [0, 1].map((k) => ({
		name: k === 0 ? "K=0（35 NFE）" : "K=1（39 NFE）",
		labels: ["200k", "210k", "220k"],
		values: [Number(get(210000, k).baseline_mean), Number(get(210000, k).candidate_mean), Number(get(220000, k).candidate_mean)],
	}));
	const cw = 5.9;
	text(s, "seed 0–49,999の1範囲で比べた推移", { x: MX, y: 2.0, w: cw, h: 0.28, fontSize: 12.5, bold: true, color: NAVY });
	s.addChart(pres.charts.LINE, series, {
		x: MX, y: 2.3, w: cw, h: 4.5,
		chartColors: [THEME.colors.accent5, THEME.colors.accent2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 9,
		showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
		valAxisMinVal: 1.75, valAxisMaxVal: 2.1, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
		valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
		valAxisLabelColor: MUTED, catAxisLabelColor: MUTED,
		showValAxisTitle: true, valAxisTitle: "FID-50k（低いほど良い）", valAxisTitleFontSize: 11, valAxisTitleColor: MUTED, valAxisTitleFontFace: "+mn-lt",
		showCatAxisTitle: true, catAxisTitle: "学習量（kimg）", catAxisTitleFontSize: 11, catAxisTitleColor: MUTED, catAxisTitleFontFace: "+mn-lt",
		valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
	});

	const tx = MX + cw + 0.3, tw = W - MX - tx;
	const cond = Object.fromEntries(POST200K_COND.map((r) => [r.condition, r]));
	const gain = Object.fromEntries(POST200K_GAIN.map((r) => [r.checkpoint, r]));
	const threeRange = (ckpt) => [
		{ text: ckpt, options: rowLabel() },
		meanSd(cond[`${ckpt}-k0`].mean, cond[`${ckpt}-k0`].std),
		meanSd(cond[`${ckpt}-k1`].mean, cond[`${ckpt}-k1`].std),
		{ text: `${fmtGain(Number(gain[ckpt].mean))} ± ${Number(gain[ckpt].std).toFixed(3)}`, options: { bold: true, color: GREEN } },
		`${gain[ckpt].positive_ranges} / ${gain[ckpt].n}`,
	];
	const [k0_220, k1_220] = [series[0].values[2], series[1].values[2]];
	assert.ok(k1_220 > k0_220);
	const oneRange = { color: ORANGE };
	const rows = [
		["学習量", "K=0のFID", "K=1のFID", "K=1の改善幅", "改善範囲"].map((c) => ({ text: c, options: head(11) })),
		threeRange("200k"),
		threeRange("210k"),
		[{ text: "220k*", options: rowLabel() }, { text: k0_220.toFixed(3), options: oneRange }, { text: k1_220.toFixed(3), options: oneRange },
			{ text: fmtGain(k0_220 - k1_220), options: { bold: true, color: RED } }, { text: "0 / 1", options: oneRange }],
	];
	s.addTable(rows, {
		x: tx, y: 2.05, w: tw, colW: [0.7, 1.3, 1.3, 1.5, tw - 4.8], rowH: [0.42, 0.45, 0.45, 0.45],
		fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: [0.02, 0.04, 0.02, 0.04], border: BORDER(),
	});
	text(s, "200k・210kは3シード範囲の平均 ± 標本SD。*220kは1シード範囲のみ。", { x: tx, y: 3.88, w: tw, h: 0.26, fontSize: 10.5, color: MUTED });

	const delta0 = Number(cond["210k-k0"].mean) - Number(cond["200k-k0"].mean);
	card(s, tx, 4.3, tw, 2.35, TINT, "result");
	chip(s, "結果", tx + 0.2, 4.45, 0.8, RED);
	text(s, [
		{ text: `継続学習でFIDは悪化した（K=0、3範囲平均で200k→210kは${fmtGain(delta0)}）`, options: { bullet: true, breakLine: true } },
		{ text: "200k・210kでは、K=1の小さな改善が3範囲とも再現した", options: { bullet: true, breakLine: true } },
		{ text: "220kではK=1もK=0より悪かったが、1範囲のK=0・K=1だけの観測", options: { bullet: true } },
	], { x: tx + 0.2, y: 4.85, w: tw - 0.4, h: 1.7, fontSize: 13, paraSpaceAfter: 5 });

	s.addNotes([
		"200kからさらに学習を続けた場合です。学習率0.001のまま継続し、2.5kごとに保存しながら220kまで学習しました。",
		`前回は1シード範囲だけでしたが、今回200kと210kを3範囲で確認しました。K=0のFIDは平均で${Number(cond["200k-k0"].mean).toFixed(3)}から${Number(cond["210k-k0"].mean).toFixed(3)}へ悪化しています。K=1の改善幅は200kで${fmtGain(Number(gain["200k"].mean))}、210kで${fmtGain(Number(gain["210k"].mean))}で、どちらも3範囲すべてで改善でした。`,
		`220kは1範囲だけで、K=0が${k0_220.toFixed(3)}、K=1が${k1_220.toFixed(3)}と大きく悪化し、K=1の方が悪い結果でした。これはK=0とK=1だけの観測なので、ほかのKやほかのモデルにまでは広げません。`,
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 8. 30,000-image training

pres.addSection({ title: "学習データ30,000枚" });
{
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "学習データ30,000枚" });
	s.addText("学習データ30,000枚での再学習", { placeholder: "title" });
	infoLine(s, "条件", "CIFAR-10の50,000枚から固定seedで30,000枚を抽出。DDPM++・EDM、学習率0.001、EMA 0.5 Mimgは全データ学習と同じ", 1.13);
	infoLine(s, "評価", "通常ODE、18 steps（35 NFE）、FID-50k（seed 0–49,999）。10kごとに100kまで", 1.53);

	const cw = 8.2;
	s.addChart(pres.charts.LINE, [{ name: "FID-50k", labels: SUBSET_TRAIN.map((r) => `${r.kimg / 1000}k`), values: SUBSET_TRAIN.map((r) => r.fid) }], {
		x: MX, y: 1.95, w: cw, h: 4.05,
		chartColors: [THEME.colors.accent1], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 9, showLegend: false,
		valAxisMinVal: 2, valAxisMaxVal: 6, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0",
		valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
		valAxisLabelColor: MUTED, catAxisLabelColor: MUTED,
		showValAxisTitle: true, valAxisTitle: "FID-50k（低いほど良い）", valAxisTitleFontSize: 11, valAxisTitleColor: MUTED, valAxisTitleFontFace: "+mn-lt",
		showCatAxisTitle: true, catAxisTitle: "学習量（kimg）", catAxisTitleFontSize: 11, catAxisTitleColor: MUTED, catAxisTitleFontFace: "+mn-lt",
		valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
	});

	const firstW = 0.75, kimgW = (cw - firstW) / SUBSET_TRAIN.length;
	s.addTable([
		[{ text: "学習量", options: head(10.5) }].concat(SUBSET_TRAIN.map((r) => ({ text: `${r.kimg / 1000}k`, options: head(10.5) }))),
		[{ text: "FID", options: rowLabel() }].concat(SUBSET_TRAIN.map((r) => (r === subsetBest ? { text: r.fid.toFixed(3), options: { bold: true, color: GREEN } } : r.fid.toFixed(3)))),
	], {
		x: MX, y: 6.08, w: cw, colW: [firstW].concat(SUBSET_TRAIN.map(() => kimgW)), rowH: 0.3,
		fontSize: 10.5, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.02, 0.01, 0.02], border: BORDER(),
	});

	const tx = MX + cw + 0.3, tw = W - MX - tx;
	[[`${subsetBest.kimg / 1000}kで最良`, subsetBest.fid.toFixed(3), GREEN, 2.05], [`${subsetLast.kimg / 1000}k（最終）`, subsetLast.fid.toFixed(3), RED, 3.5]].forEach(([label, value, color, y]) => {
		card(s, tx, y, tw, 1.25, TINT, `stat ${label}`);
		text(s, label, { x: tx + 0.25, y: y + 0.15, w: tw - 0.5, h: 0.3, fontSize: 13, color: MUTED });
		text(s, value, { x: tx + 0.25, y: y + 0.45, w: tw - 0.5, h: 0.7, fontSize: 34, bold: true, color, valign: "middle" });
	});
	text(s, [
		{ text: "50k以降はFIDが悪化し続ける（過学習的な挙動）", options: { bullet: true, breakLine: true } },
		{ text: "40kは前後より悪く、推移の中で不規則な点", options: { bullet: true } },
	], { x: tx, y: 4.95, w: tw, h: 1.7, fontSize: 13, paraSpaceAfter: 6 });

	s.addNotes([
		"ここからは、使える学習データが少ない場合です。CIFAR-10の50,000枚から固定seedで30,000枚を抽出し、モデルや学習率などは全データ学習と同じ条件で100kまで学習しました。",
		`FID-50kは${subsetBest.kimg / 1000}kの${subsetBest.fid.toFixed(3)}が最良で、その後は悪化を続け、100kでは${subsetLast.fid.toFixed(3)}になりました。データが少ないため、過学習的な挙動だと解釈しています。40kは前後より悪く、不規則な点です。`,
		"次のページから、このモデルでRestartが効くかを見ます。",
	].join("\n\n"));
}

// ---------------------------------------------------------------------------- 9-10. K sweeps on the 30,000-image model

// Both panels are native PowerPoint objects so they can be moved and edited on the slide.
const SUBSET_NEAR_GAP = 0.02;
const SUBSET_DELTA_MAX = Math.max(...[SUBSET_MAIN18_FID, SUBSET_NFE55_FID].flatMap((table) =>
	SUBSET_CKPTS.flatMap((ckpt) => Object.values(table[ckpt]).map((fid) => Math.abs(fid - table[ckpt][0])))));

// Blue-white-red, shared between the two sweeps.
function heatCell(delta) {
	const t = Math.max(-1, Math.min(1, delta / SUBSET_DELTA_MAX));
	const fade = Math.round(255 * (1 - Math.abs(t))).toString(16).padStart(2, "0").toUpperCase();
	const fill = t < 0 ? `${fade}${fade}FF` : `FF${fade}${fade}`;
	const label = delta === 0 ? "0" : `${delta > 0 ? "+" : "−"}${Math.abs(delta).toFixed(2)}`;
	return { text: label, options: { fill: { color: fill }, color: Math.abs(t) > 0.6 ? C.background1 : C.text1 } };
}

function bestKChart(slide, table, x, y, w, h, name) {
	const stats = SUBSET_CKPTS.map((ckpt) => {
		const fids = table[ckpt];
		const best = Object.keys(fids).map(Number).reduce((a, b) => (fids[b] < fids[a] ? b : a));
		const near = Object.keys(fids).map(Number).filter((k) => fids[k] - fids[best] <= SUBSET_NEAR_GAP);
		return { kimg: parseInt(ckpt, 10), best, near, mean: meanK(fids) };
	});
	const extra = stats.flatMap((r) => r.near.filter((k) => k !== r.best).map((k) => ({ kimg: r.kimg, k })));
	const blanks = (n) => Array(n).fill(null);
	const series = [{ name: "学習量", values: stats.map((r) => r.kimg).concat(extra.map((r) => r.kimg)) }];
	const colors = [];
	if (extra.length) {
		series.push({ name: "近いK（差0.02以下）", values: blanks(stats.length).concat(extra.map((r) => r.k)) });
		colors.push("9ECAE1");
	}
	series.push({ name: MEAN_K_SERIES, values: stats.map((r) => r.mean).concat(blanks(extra.length)) });
	colors.push(THEME.colors.accent2);
	series.push({ name: BEST_K_SERIES, values: stats.map((r) => r.best).concat(blanks(extra.length)) });
	colors.push(THEME.colors.dk1);
	slide.addChart(pres.charts.SCATTER, series, {
		x, y, w, h, objectName: name,
		chartColors: colors, lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 10,
		showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
		valAxisMinVal: 0, valAxisMaxVal: 8, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0",
		catAxisMinVal: 10, catAxisMaxVal: 70, catAxisMajorUnit: 10,
		valAxisLabelFontSize: 11, catAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
		valAxisLabelColor: MUTED, catAxisLabelColor: MUTED,
		showValAxisTitle: true, valAxisTitle: "K", valAxisTitleFontSize: 11, valAxisTitleColor: MUTED, valAxisTitleFontFace: "+mn-lt",
		showCatAxisTitle: true, catAxisTitle: "学習量（kimg）", catAxisTitleFontSize: 11, catAxisTitleColor: MUTED, catAxisTitleFontFace: "+mn-lt",
		valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
	});
}

function subsetSlide({ title, condition, table, results, gainNote, notes }) {
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "学習データ30,000枚" });
	s.addText(title, { placeholder: "title" });
	infoLine(s, "条件", condition, 1.13);

	const lw = 7.1, top = 1.62, firstW = 0.8, kW = (lw - firstW) / 9;
	text(s, "FID(K) − FID(K=0)（青は改善、赤は悪化）", { x: MX, y: top, w: lw, h: 0.28, fontSize: 12.5, bold: true, color: NAVY });
	const ks = [...Array(9).keys()];
	s.addTable([
		[{ text: "", options: head(11) }].concat(ks.map((k) => ({ text: `K=${k}`, options: head(11) }))),
	].concat(SUBSET_CKPTS.map((ckpt) => [{ text: ckpt, options: Object.assign(rowLabel(), { align: "center" }) }].concat(ks.map((k) => heatCell(table[ckpt][k] - table[ckpt][0]))))), {
		x: MX, y: top + 0.32, w: lw, colW: [firstW].concat(ks.map(() => kW)), rowH: [0.36, 0.62, 0.62, 0.62],
		fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.02, 0.01, 0.02], border: BORDER(), objectName: "heatmap FID delta",
	});

	const sy = 4.4;
	s.addTable([["checkpoint", "K=0のFID", "最良K", "改善幅"].map((c) => ({ text: c, options: head(11) }))].concat(results.map((r) => [
		r.ckpt, table[r.ckpt][0].toFixed(3), `K=${r.k}`,
		r.k === 0 ? { text: "なし", options: { bold: true, color: RED } } : gainCell(r.gain),
	])), {
		x: MX, y: sy, w: lw, colW: [0.28, 0.26, 0.21, 0.25].map((w) => w * lw), rowH: 0.36,
		fontSize: 12, color: C.text1, align: "center", valign: "middle", margin: [0.01, 0.04, 0.01, 0.04], border: BORDER(), objectName: "best K summary",
	});
	text(s, `改善幅 = FID(K=0) − 最小FID。${gainNote}40k・60kはK=1〜8のすべてがK=0より悪い。`,
		{ x: MX, y: sy + 0.36 * 4 + 0.12, w: lw, h: 0.5, fontSize: 11, color: MUTED });

	const rx = MX + lw + 0.3, rw = W - MX - rx;
	text(s, "最良K、近いK、その単純平均", { x: rx, y: top, w: rw, h: 0.28, fontSize: 12.5, bold: true, color: NAVY });
	bestKChart(s, table, rx, top + 0.32, rw, 4.85, "chart best K");
	s.addNotes(notes.join("\n\n"));
}

subsetSlide({
	title: "30,000枚モデルのK探索：主経路を固定（FID-50k）",
	condition: "主経路18 steps、K=0〜8、NFE = 35+4K（35〜67）、t_min = 0.14、各条件50,000枚、20k・40k・60k",
	table: SUBSET_MAIN18_FID,
	results: SUBSET_MAIN18,
	gainNote: "",
	notes: [
		"30,000枚で学習したモデルに、全データのときと同じ主経路固定のK探索を行いました。ここでは20k、40k、60kのFID-50kを示します。",
		`20kでは最良K=3で${fmtGain(SUBSET_MAIN18[0].gain)}改善しました。一方、40kと60kではK=1〜8のすべてがK=0より悪く、Kを増やすほど悪化しています。`,
		"全データでは最後までK=0より良いKがありましたが、30,000枚ではそうなっていません。学習不足の段階では改善し、十分に学習すると悪化する傾向です。40kは学習の推移でも不規則な点でした。今回の条件での結果で、データ不足全般の結論ではありません。",
	],
});

subsetSlide({
	title: "30,000枚モデルのK探索：計算量を固定（FID-50k）",
	condition: "NFE = 55で固定、主経路28−2K steps、K=0〜8、t_min = 0.14、各条件50,000枚、20k・40k・60k",
	table: SUBSET_NFE55_FID,
	results: SUBSET_NFE55,
	gainNote: "K=0は同じ55 NFEの通常ODE（28 steps）。",
	notes: [
		"同じ30,000枚モデルで、計算量を55 NFEに固定した探索です。こちらも20k、40k、60kのFID-50kを示します。",
		`傾向は主経路固定と同じで、20kでは最良K=6で同じ計算量のODEより${fmtGain(SUBSET_NFE55[0].gain)}改善しましたが、40kと60kではK=1〜8のすべてがK=0より悪くなりました。`,
		"ヒートマップの色の範囲は前のページとそろえています。",
	],
});

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

// pptxgenjs sets line, marker size and the X axis format for all scatter series at once. Keep the
// line on the mean K only, and size the markers so the best K stays visible under the mean.
async function styleScatter(file) {
	const zip = await JSZip.loadAsync(fs.readFileSync(file));
	for (const name of Object.keys(zip.files).filter((f) => /^ppt\/charts\/chart\d+\.xml$/.test(f))) {
		const xml = await zip.file(name).async("string");
		if (!xml.includes("<c:scatterChart>")) continue;
		zip.file(name, xml.split("<c:ser>").map((part, i) => {
			if (i === 0) return part;
			if (part.includes(`<c:v>${MEAN_K_SERIES}</c:v>`)) return part.replace('<c:size val="10"/>', '<c:size val="6"/>');
			const size = part.includes(`<c:v>${BEST_K_SERIES}</c:v>`) ? 13 : 10;
			return part.replace(/<a:ln w="\d+"[^>]*>.*?<\/a:ln>/, "<a:ln><a:noFill/></a:ln>").replace('<c:size val="10"/>', `<c:size val="${size}"/>`);
		}).join("<c:ser>").replace('<c:numFmt formatCode="0" sourceLinked="1"/>', '<c:numFmt formatCode="0&quot;k&quot;" sourceLinked="0"/>'));
	}
	fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
	await pres.writeFile({ fileName: OUT });
	await applyTheme(OUT, THEME);
	await setEastAsianFont(OUT);
	await styleScatter(OUT);
	console.log(`wrote ${OUT}`);
	console.log("fixed main18 FID-50k:", MAIN18.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log("fixed NFE55 FID-50k:", NFE55.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log("3-seed gain:", SEEDS3_GAIN.map((r) => `${r.checkpoint} ${fmtGain(Number(r.mean))}`).join(", "));
	console.log("30k main18:", SUBSET_MAIN18.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
	console.log("30k NFE55:", SUBSET_NFE55.map((r) => `${r.ckpt} K=${r.k} ${fmtGain(r.gain)}`).join(", "));
})();
