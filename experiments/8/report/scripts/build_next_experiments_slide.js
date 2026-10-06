// Build the one-slide "今後の実験予定" deck that sits beside the progress report.
//
// Usage (from the EDM repository root):
//   NODE_PATH=<dir with pptxgenjs and jszip> PPTX_SKILL_DIR=<pptx skill dir> \
//     node experiments/8/report/scripts/build_next_experiments_slide.js
//
// The saved-checkpoint count is read from the retraining run directory at build time.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const REPORT_DIR = path.join("experiments", "8", "report");
const OUT = path.join(REPORT_DIR, "restart_sampling_next_experiments.pptx");
const RUN_DIR = path.join("experiments", "0", "1_cifar10_cond_edm_10k_retraining", "training-runs", "cifar10-cond-ddpmpp-edm-seed0-10k");
const TOTAL_CHECKPOINTS = 20;

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

// Latest saved checkpoint of the exact-10k retraining run.
const snapshots = fs.readdirSync(RUN_DIR).filter((n) => /^network-snapshot-\d{6}\.pkl$/.test(n)).sort();
if (snapshots.length === 0) throw new Error(`no network snapshots in ${RUN_DIR}`);
const savedLast = `${parseInt(snapshots[snapshots.length - 1].slice(17, 23), 10) / 1000}k`;
const savedCount = snapshots.length;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "今後の実験予定";
const C = pres.SchemeColor;
const ORANGE = C.accent2, NAVY = C.text2, TINT = C.background2;

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
		shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.17, x, y, w, h: 0.34,
		fill: { color: fill || NAVY }, line: { type: "none" }, color: textColor || C.background1,
		fontSize: 13, bold: true, align: "center", valign: "middle", margin: 0, objectName: `chip ${label}`,
	});
}

pres.addSection({ title: "まとめ" });
const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "まとめ" });
s.addText("今後の実験予定", { placeholder: "title" });

const groups = [
	{ label: "進行中", color: ORANGE, x: MX, items: [
		["10k間隔の厳密再学習", `10k〜${savedLast}を保存済み（${savedCount}/${TOTAL_CHECKPOINTS}点）。200kまで学習を継続中`],
		["FID-5kでの最良Kの走査", "10k〜100k、K=0〜8で評価。最良Kは学習段階ごとに変動している"],
	] },
	{ label: "実験予定", color: NAVY, x: 7.03, items: [
		["FID-25kでの最良Kの走査", "評価画像を5,000→25,000枚に増やし、FIDのばらつきの影響を減らして最良Kを確認する"],
		["少量サンプル（50k→30k）で再学習", "学習データを50,000→30,000枚に減らして再学習し、Restartの効果との関係を調べる"],
	] },
];
const colW = 5.75, cardH = 2.1, cardY = [1.95, 4.3];

let n = 0;
groups.forEach(({ label, color, x, items }) => {
	chip(s, label, x, 1.3, 1.25, color);
	items.forEach(([title, desc], i) => {
		n += 1;
		const y = cardY[i];
		card(s, x, y, colW, cardH, TINT, `${label} ${n}`);
		s.addText(String(n), { shape: pres.shapes.OVAL, x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, fill: { color }, line: { type: "none" }, color: C.background1, fontSize: 17, bold: true, align: "center", valign: "middle", margin: 0, objectName: `number ${n}` });
		text(s, title, { x: x + 1.0, y: y + 0.33, w: colW - 1.25, h: 0.5, fontSize: 18.5, bold: true, color: NAVY, valign: "middle" });
		text(s, desc, { x: x + 1.0, y: y + 1.0, w: colW - 1.25, h: 0.95, fontSize: 15 });
	});
});

// Flow from the running work to the planned experiments.
s.addShape(pres.shapes.RIGHT_ARROW, { x: MX + colW + 0.12, y: 3.95, w: 0.48, h: 0.55, fill: { color: C.accent5 }, line: { type: "none" }, objectName: "flow arrow" });

s.addNotes([
	`進行中の実験です。1つ目は10k間隔の厳密な再学習で、10k〜${savedLast}まで保存済み（${savedCount}/${TOTAL_CHECKPOINTS}点）、200kまで学習を続けています。2つ目はFID-5kでの最良Kの走査で、10k〜100kについてK=0〜8を評価し、最良Kが学習段階ごとに変わることを確認しました。`,
	"今後の予定です。3つ目として、FID-5kは評価画像が少なくばらつきの影響が大きいため、FID-25kで最良Kを走査し直し、最良Kを確認します。",
	"4つ目として、学習データを50,000枚から30,000枚に減らして再学習し、学習データ量とRestartの効果の関係を調べます。",
].join("\n\n"));

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
	console.log(`wrote ${OUT} (saved through ${savedLast}, ${savedCount}/${TOTAL_CHECKPOINTS})`);
})();
