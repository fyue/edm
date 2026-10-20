#!/usr/bin/env python3
"""CPU tests for compatible-result merging and the 8-11/8-12 comparison style."""

import csv
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import plot_comparison as plotting


class PlotComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="edm-plot-comparison-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ, MPLCONFIGDIR=str(self.root / "matplotlib"))
        environment.start()
        self.addCleanup(environment.stop)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, "all")

    def row(self, checkpoint="20k", count=5000, k=0, fid=2.0, **updates):
        row = dict(
            condition=f"{checkpoint}-k{k}", checkpoint=checkpoint, num_images=str(count),
            seeds=f"0-{count - 1}", restart_k=str(k), main_steps="18",
            restart_steps="3" if k else "0", t_min="0.06" if k else "0",
            t_max="0.30" if k else "0", nfe=str(35 + 4 * k), fid=str(fid),
            network=f"{plotting.NETWORK_DIR}/network-snapshot-{int(checkpoint[:-1]) * 1000:06d}.pkl",
            fid_ref=plotting.FID_REF,
        )
        row.update(updates)
        return row

    def summary(self, name, rows):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_source_priority_aliases_and_ode_baseline(self):
        primary = self.summary("primary.tsv", [self.row(k=1, fid=2.0)])
        fallback = self.summary("fallback.tsv", [self.row(k=1, fid=1.0), self.row(k=5)])
        baseline_row = self.row(checkpoint="200k", count=50000)
        baseline_row["network"] = "/different/pc/edm/" + baseline_row["network"]
        baseline_row["fid_ref"] = "/different/pc/edm/" + baseline_row["fid_ref"]
        baseline_row["t_min"] = "0.14"
        baseline = self.summary("baseline.tsv", [baseline_row])
        groups = plotting.load_groups([
            ("8-4", [primary]), ("8-6", [fallback]), ("8-11", [baseline]),
        ])
        self.assertEqual(groups["20k", 5000][1]["fid"], 2.0)
        self.assertEqual(groups["20k", 5000][1]["source_experiment"], "8-4")
        self.assertEqual(groups["20k", 5000][5]["source_experiment"], "8-6")
        row = groups["200k", 50000][0]
        self.assertEqual(row["source_summary"], str(baseline))
        self.assertEqual(row["source_line"], 2)
        self.assertEqual(row["t_min"], 0)

    def test_duplicate_within_experiment_is_an_error(self):
        first = self.summary("first.tsv", [self.row()])
        second = self.summary("second.tsv", [self.row(fid=2.1)])
        with self.assertRaisesRegex(ValueError, "Duplicate compatible result in 8-4"):
            plotting.load_groups([("8-4", [first, second])])
        same_file = self.summary("same.tsv", [self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "Duplicate compatible result"):
            plotting.load_groups([("8-4", [same_file])])

    def test_incompatible_sampling_protocols_are_excluded(self):
        cases = [
            dict(checkpoint="10k"), dict(main_steps="28"), dict(t_min="0.14"),
            dict(t_max="0.4"), dict(restart_steps="4"), dict(nfe="55"),
            dict(network="different-training/network-snapshot-020000.pkl"),
            dict(network=f"{plotting.NETWORK_DIR}/network-snapshot-040000.pkl"),
            dict(seeds="5000-9999"), dict(fid_ref="fid-refs/other.npz"),
            dict(rho="8"), dict(restart_noise="2"), dict(S_churn="1"),
            dict(gen_batch="32"), dict(fid_batch="32"),
        ]
        for updates in cases:
            with self.subTest(updates=updates):
                row = self.row(k=1)
                row.update(updates)
                self.assertIsNone(plotting.compatible_row(row))
        for invalid in ("nan", "inf", "-1"):
            with self.assertRaisesRegex(ValueError, "Invalid FID"):
                plotting.compatible_row(self.row(fid=invalid))

    def test_color_scale_overflow_missing_cells_and_ticks(self):
        import numpy as np
        from matplotlib.colors import to_rgba
        groups = {("20k", 5000): {0: {"fid": 2.0}, 1: {"fid": 1.5}, 2: {"fid": 2.5}},
                  ("40k", 5000): {1: {"fid": 2.0}}}
        figure = plotting.make_figure(groups, 5000)
        heatmap = figure.axes[1].images[0]
        self.assertEqual(heatmap.cmap(heatmap.norm(0)), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(heatmap.cmap(heatmap.norm(-0.4)), to_rgba("#3b4cc0"))
        self.assertEqual(heatmap.cmap(heatmap.norm(0.4)), to_rgba("#b40426"))
        self.assertEqual(heatmap.cmap(heatmap.norm(-0.5)), heatmap.cmap(heatmap.norm(-0.4)))
        self.assertEqual(heatmap.colorbar.extend, "both")
        self.assertAlmostEqual(float(heatmap.get_array()[0, 1]), -0.5)
        self.assertTrue(np.ma.getmaskarray(heatmap.get_array())[1].all())
        self.assertEqual(tuple(heatmap.cmap.get_bad()), to_rgba(plotting.MISSING_COLOR))
        np.testing.assert_allclose(heatmap.colorbar.get_ticks(), np.linspace(-0.4, 0.4, 9))
        np.testing.assert_array_equal(figure.axes[0].get_xticks(), range(9))
        np.testing.assert_array_equal(figure.axes[1].get_xticks(), range(9))
        np.testing.assert_array_equal(figure.axes[2].get_xticks(), range(20, 201, 20))
        np.testing.assert_array_equal(figure.axes[2].get_yticks(), range(9))
        self.assertTrue(np.isnan(figure.axes[0].lines[0].get_ydata()[3:]).all())
        self.assertTrue(np.isnan(figure.axes[2].lines[0].get_ydata()).all())
        self.assertEqual(figure.axes[2].texts[0].get_text(), "No complete K sweep")

    def test_checkpoint_colors_ties_and_missing_checkpoint_break(self):
        import numpy as np
        complete = {k: {"fid": 2.0 if k in (2, 3) else 2.1} for k in range(9)}
        groups = {("40k", 25000): complete, ("80k", 25000): complete}
        figure = plotting.make_figure(groups, 25000)
        color = figure.axes[0].lines[0].get_color()
        best = figure.axes[2].lines[0].get_ydata()
        self.assertEqual(best[1], 2)
        self.assertTrue(np.isnan(best[2]))
        self.assertEqual(best[3], 2)
        self.assertEqual(figure.axes[1].images[0].colorbar.extend, "neither")
        groups["20k", 25000] = {0: {"fid": 2.0}, 8: {"fid": 1.5}}
        figure = plotting.make_figure(groups, 25000)
        self.assertEqual(color, figure.axes[0].lines[1].get_color())
        self.assertEqual(figure.axes[1].images[0].colorbar.extend, "min")

    def test_outputs_and_input_preservation_and_stage_selection(self):
        rows = [self.row(count=count, k=k) for count in plotting.STAGES for k in plotting.KS]
        summary = self.summary("source.tsv", rows)
        before = hashlib.sha256(summary.read_bytes()).hexdigest()
        sources = [("8-4", [summary])]
        outdir = self.root / "all-plots"
        plotting.main(["--outdir", str(outdir)], sources=sources)
        for count in plotting.STAGES:
            for extension in ("png", "svg"):
                self.assertTrue((outdir / f"fixed_main18_tmin006_fid{count}.{extension}").is_file())
        with (outdir / "comparison_fid_summary.tsv").open() as handle:
            exported = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(exported), 27)
        self.assertTrue(all(row["source_summary"] == str(summary) for row in exported))
        self.assertEqual(before, hashlib.sha256(summary.read_bytes()).hexdigest())
        selected = self.root / "one-stage"
        plotting.main(["--fid-stage", "50000", "--outdir", str(selected)], sources=sources)
        self.assertEqual(len(list(selected.glob("*.png"))), 1)
        with (selected / "comparison_fid_summary.tsv").open() as handle:
            self.assertEqual({row["num_images"] for row in csv.DictReader(handle, delimiter="\t")}, {"50000"})

    def test_merged_baseline_table_is_not_double_counted_with_its_runs(self):
        baseline = self.root / "experiments/8/11_restart_fixed_main18_tmin014_k_sweep"
        merged = baseline / "results/restart_fid_summary.tsv"
        run = baseline / "runs/20k-140k/results/restart_fid_summary.tsv"
        for path in (merged, run):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with mock.patch.object(plotting, "REPO_ROOT", self.root):
            self.assertEqual(plotting.default_sources()[-1], ("8-11", [merged]))
            merged.unlink()
            self.assertEqual(plotting.default_sources()[-1], ("8-11", [run]))


if __name__ == "__main__":
    unittest.main()
