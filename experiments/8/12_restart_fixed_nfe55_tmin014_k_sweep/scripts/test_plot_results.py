#!/usr/bin/env python3
"""CPU checks for the fixed comparison scale and partial-result plotting."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import plot_results as plotting


class PlotResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = mock.patch.dict(os.environ, MPLCONFIGDIR=str(self.root / "matplotlib"))
        self.environment.start()
        self.addCleanup(self.environment.stop)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, "all")

    def test_scale_overflow_missing_cells_and_ticks(self):
        import numpy as np
        from matplotlib.colors import to_rgba
        groups = {("20k", 5000): {0: {"fid": 2.0}, 1: {"fid": 1.5}, 2: {"fid": 2.5}},
                  ("40k", 5000): {1: {"fid": 2.0}}}
        figure = plotting.make_figure(groups, 5000)
        image = figure.axes[1].images[0]
        self.assertEqual(image.cmap(image.norm(0)), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(image.cmap(image.norm(-0.4)), to_rgba("#3b4cc0"))
        self.assertEqual(image.cmap(image.norm(0.4)), to_rgba("#b40426"))
        self.assertEqual(image.cmap(image.norm(-0.5)), image.cmap(image.norm(-0.4)))
        self.assertEqual(image.colorbar.extend, "both")
        self.assertAlmostEqual(float(image.get_array()[0, 1]), -0.5)
        self.assertTrue(np.ma.getmaskarray(image.get_array())[1].all())
        self.assertEqual(tuple(image.cmap.get_bad()), to_rgba(plotting.MISSING_COLOR))
        np.testing.assert_array_equal(figure.axes[0].get_xticks(), range(9))
        np.testing.assert_array_equal(figure.axes[1].get_xticks(), range(9))
        np.testing.assert_array_equal(figure.axes[2].get_xticks(), range(20, 201, 20))
        np.testing.assert_array_equal(figure.axes[2].get_yticks(), range(9))
        self.assertTrue(np.isnan(figure.axes[2].lines[0].get_ydata()).all())
        np.testing.assert_allclose(image.colorbar.get_ticks(), np.linspace(-0.4, 0.4, 9))
        self.assertEqual(image.colorbar.formatter(0.2), "0.2")

    def test_checkpoint_colors_optimal_ties_and_one_sided_overflow(self):
        import numpy as np
        complete = {k: {"fid": 2.0 if k in (2, 3) else 2.1} for k in range(9)}
        groups = {("40k", 25000): complete}
        figure = plotting.make_figure(groups, 25000)
        color = figure.axes[0].lines[0].get_color()
        best = figure.axes[2].lines[0].get_ydata()
        self.assertEqual(best[1], 2)
        self.assertTrue(np.isnan(best[0]))
        self.assertEqual(figure.axes[1].images[0].colorbar.extend, "neither")
        groups["20k", 25000] = {0: {"fid": 2.0}, 8: {"fid": 1.5}}
        figure = plotting.make_figure(groups, 25000)
        self.assertEqual(color, figure.axes[0].lines[1].get_color())
        self.assertEqual(figure.axes[1].images[0].colorbar.extend, "min")

    def test_complete_and_partial_summary_cli_preserves_inputs(self):
        experiment = plotting.experiment
        conditions = experiment.load_conditions(
            experiment.DEFAULT_ROOT / "config/restart_fixed_nfe55_tmin014.tsv")
        args = SimpleNamespace(fid_ref=self.root / "ref", gen_batch=64, fid_batch=64)
        rows = []
        with mock.patch.object(experiment, "identity", return_value={"size": 1, "sha256": "a" * 64}):
            for count in experiment.STAGES:
                for row in conditions:
                    protocol = experiment.protocol(row, args)
                    rows.append(dict(row, num_images=count, seeds=f"0-{count - 1}", fid=2.0,
                                     generate_sec=1.0, fid_sec=1.0, reused_images=0, reuse_source="",
                                     images="/fixture/images", log="/fixture/log", fid_ref="/fixture/ref",
                                     gen_batch=64, fid_batch=64, rho=7, restart_noise=1,
                                     protocol_id=experiment.digest(protocol), protocol=protocol,
                                     gpu_name="CPU fixture", generated_images=count, elapsed_sec=2.0))
        summary = self.root / "summary.json"
        table = self.root / "table.tsv"
        table.write_text("unchanged table\n")
        command = [sys.executable, str(Path(plotting.__file__)), "--summary", str(summary),
                   "--outdir", str(self.root / "plots")]
        for data, stages in ((rows, experiment.STAGES), ([rows[1]], (5000,))):
            summary.write_text(json.dumps(data))
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (summary, table)]
            groups = plotting.load_groups(summary)
            self.assertEqual(len(groups), 30 if len(data) == 270 else 1)
            output = subprocess.check_output(command, text=True)
            for count in stages:
                for extension in ("png", "svg"):
                    name = f"fixed_nfe55_tmin014_fid{count}.{extension}"
                    self.assertIn(name, output)
                    self.assertTrue((self.root / "plots" / name).is_file())
            after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (summary, table)]
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
