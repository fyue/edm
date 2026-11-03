"""CPU checks for selected-checkpoint FID stage figures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np

import plot_main18_fid5k as plot


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.results = [dict(checkpoint=checkpoint, condition=f"{checkpoint}-k{k}", restart_k=k,
                             num_images=5000, seeds="0-4999", main_steps=18,
                             fid=5.0 + (-0.02 if checkpoint == "20k" else 0.04) * k)
                        for checkpoint in plot.CHECKPOINTS for k in plot.KS]

    def test_checkpoint_filter_and_exact_white(self):
        rows = self.results + [dict(self.results[0], checkpoint="80k", condition="80k-k0", fid=1e6),
                               dict(self.results[0], checkpoint="100k", condition="100k-k0", fid=2e6)]
        figure = plot.build_figure(rows)
        self.addCleanup(plot.plt.close, figure)
        left, middle, right = figure.axes[:3]
        self.assertEqual([line.get_label() for line in left.lines], list(plot.CHECKPOINTS))
        self.assertEqual([label.get_text() for label in middle.get_yticklabels()], list(plot.CHECKPOINTS))
        np.testing.assert_array_equal(right.lines[0].get_xdata(), [20, 40, 60])
        heatmap = middle.images[0]
        self.assertEqual(heatmap.get_array().shape, (3, 9))
        self.assertEqual(tuple(heatmap.cmap(heatmap.norm(0))), (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(heatmap.norm.vmax, 0.32)
        self.assertEqual(heatmap.norm.vmin, -heatmap.norm.vmax)
        self.assertLess(heatmap.cmap(heatmap.norm(-0.1))[0], heatmap.cmap(heatmap.norm(-0.1))[2])
        self.assertGreater(heatmap.cmap(heatmap.norm(0.1))[0], heatmap.cmap(heatmap.norm(0.1))[2])
        self.assertIn(0.0, figure.axes[3].get_yticks())
        self.assertIn("0.0", [label.get_text() for label in figure.axes[3].get_yticklabels()])

    def test_missing_duplicate_invalid_and_wrong_stage(self):
        for rows in (self.results[:-1], self.results + [self.results[0]]):
            with self.assertRaises(ValueError):
                plot.build_figure(rows)
        for field, invalid in (("fid", float("nan")), ("fid", float("inf")), ("fid", -1),
                               ("restart_k", 0.5), ("seeds", "1-5000"), ("main_steps", 28),
                               ("condition", "other"), ("num_images", 25000)):
            rows = copy.deepcopy(self.results)
            rows[0][field] = invalid
            with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                plot.build_figure(rows)

    def test_all_zero_and_one_sided_values(self):
        for direction in (0, -1, 1):
            rows = [dict(row, fid=5.0 + direction * row["restart_k"] * 0.01) for row in self.results]
            figure = plot.build_figure(rows)
            heatmap = figure.axes[1].images[0]
            self.assertEqual(tuple(heatmap.cmap(heatmap.norm(0))), (1.0, 1.0, 1.0, 1.0))
            self.assertGreater(heatmap.norm.vmax, 0)
            plot.plt.close(figure)

    def test_redraw_only_two_outputs_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in self.results:
                (root / f"{row['condition']}-fid5000.json").write_text(json.dumps(row))
            unrelated = root / "fixed_main18_tmin014_fid25000.png"
            unrelated.write_bytes(b"do not change")
            before = {path: path.read_bytes() for path in root.iterdir()}
            plot.redraw(root)
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertEqual({p.name for p in root.iterdir()} - {p.name for p in before},
                             {f"{plot.figure_name(5000)}.png", f"{plot.figure_name(5000)}.svg"})
            with Image.open(root / f"{plot.figure_name(5000)}.png") as image:
                self.assertEqual(image.size, (3240, 900))
            tree = ET.parse(root / f"{plot.figure_name(5000)}.svg")
            texts = " ".join(node.text or "" for node in tree.iter("{http://www.w3.org/2000/svg}text"))
            for checkpoint in plot.CHECKPOINTS:
                self.assertIn(checkpoint, texts)
            for checkpoint in ("80k", "100k"):
                self.assertNotIn(checkpoint, texts)

    def test_25k_and_50k_filter_stage_and_preserve_other_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for extension in ("png", "svg"):
                (root / f"{plot.figure_name(5000)}.{extension}").write_bytes(b"keep 5k figure")
            for count in (25000, 50000):
                rows = [dict(row, num_images=count, seeds=f"0-{count - 1}") for row in self.results]
                for row in rows:
                    (root / f"{row['condition']}-fid{count}.json").write_text(json.dumps(row))
                figure = plot.build_figure(self.results + rows, count)
                self.assertIn(f"FID-{count}", figure.axes[0].get_title())
                np.testing.assert_array_equal(figure.axes[2].lines[0].get_xdata(), [20, 40, 60])
                heatmap = figure.axes[1].images[0]
                self.assertEqual(tuple(heatmap.cmap(heatmap.norm(0))), (1.0, 1.0, 1.0, 1.0))
                plot.plt.close(figure)
                before = {path: path.read_bytes() for path in root.iterdir()}
                plot.redraw(root, count)
                for path, contents in before.items():
                    self.assertEqual(path.read_bytes(), contents)
                tree = ET.parse(root / f"{plot.figure_name(count)}.svg")
                texts = " ".join(node.text or "" for node in tree.iter("{http://www.w3.org/2000/svg}text"))
                self.assertIn(f"FID-{count}", texts)
                self.assertNotIn("80k", texts)
                self.assertNotIn("100k", texts)
                broken = copy.deepcopy(rows)
                broken[0]["seeds"] = "0-4999"
                with self.assertRaises(ValueError):
                    plot.build_figure(broken, count)
        with self.assertRaises(ValueError):
            plot.build_figure(self.results, 10000)


if __name__ == "__main__":
    unittest.main()
