#!/usr/bin/env python3
"""CPU checks for near-optimal K distributions."""

import os
import tempfile
import unittest
from unittest import mock

import plot_results as plotting


class NearOptimalTests(unittest.TestCase):
    def test_fid_weighted_all_candidates_temperature_and_missing(self):
        import math
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        ks = list(range(9))
        self.assertAlmostEqual(plotting.fid_weighted_mean(ks, [2.0] * 9, 0.02), 4)
        fids = [2.0] * 8 + [2.1]
        expected = (28 + 8 * math.exp(-0.1)) / (8 + math.exp(-0.1))
        self.assertAlmostEqual(plotting.fid_weighted_mean(ks, fids, 1.0), expected)
        self.assertAlmostEqual(plotting.fid_weighted_mean(ks, [v + 10 for v in fids], 1.0), expected)
        unique = [2.0 if k == 7 else 3.0 for k in ks]
        self.assertAlmostEqual(plotting.fid_weighted_mean(ks, unique, 0.001), 7)
        self.assertAlmostEqual(plotting.fid_weighted_mean(ks, unique, 1e10), 4, places=8)
        for temperature in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                plotting.fid_weighted_mean(ks, fids, temperature)
        group = {k: {"fid": v} for k, v in zip(ks, fids)}
        figure = plotting.make_dots_figure(
            {("20k", 50000): group, ("60k", 50000): group}, 50000, weighted_temperature=1.0)
        try:
            np.testing.assert_allclose(figure.get_size_inches(), [14, 5.5])
            self.assertEqual(len(figure.axes), 3)
            right = figure.axes[1]
            self.assertEqual(len(right.collections), 1)
            np.testing.assert_allclose(right.collections[0].get_offsets(), [[0, 0], [2, 0]])
            np.testing.assert_allclose(right.lines[0].get_ydata()[:3],
                                       [expected, np.nan, expected], equal_nan=True)
            self.assertIn("tau=1", right.lines[0].get_label())
        finally:
            plt.close(figure)

    def test_two_panel_dots_boundaries_ties_missing_and_heatmap(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import to_rgba

        complete = {k: {"fid": 2.1} for k in range(9)}
        for k, value in {1: 2.0, 2: 2.0, 3: 2.01, 4: 2.0100001, 5: 2.02, 6: 2.0200001}.items():
            complete[k]["fid"] = value
        groups = {("20k", 50000): complete,
                  ("40k", 50000): {0: {"fid": 2.0}, 1: {"fid": 1.9}},
                  ("60k", 50000): {1: {"fid": 1.9}}}
        figure = plotting.make_dots_figure(groups, 50000)
        try:
            np.testing.assert_allclose(figure.get_size_inches(), [14, 5.5])
            self.assertEqual(len(figure.axes), 3)  # Two data panels plus one colorbar.
            left, right = figure.axes[:2]
            np.testing.assert_allclose(right.collections[0].get_offsets(), [[0, 1], [0, 2], [0, 3]])
            np.testing.assert_allclose(right.collections[1].get_offsets(), [[0, 4], [0, 5]])
            np.testing.assert_allclose(right.collections[2].get_offsets(), [[0, 1]])
            self.assertEqual(len(right.lines), 1)
            self.assertAlmostEqual(right.lines[0].get_ydata()[0], 3.0)
            self.assertTrue(np.isnan(right.lines[0].get_ydata()[1:]).all())
            self.assertEqual(len(right.patches), 9)
            self.assertTrue(all(p.get_facecolor() == to_rgba("#eeeeee") for p in right.patches))
            self.assertGreater(right.collections[2].get_sizes()[0], right.collections[0].get_sizes()[0])
            image = left.images[0]
            np.testing.assert_allclose(image.get_array()[0],
                                       [complete[k]["fid"] - 2.1 for k in range(9)])
            self.assertAlmostEqual(float(image.get_array()[1, 1]), -0.1)
            self.assertTrue(np.ma.getmaskarray(image.get_array())[2].all())
            self.assertEqual((image.norm.vmin, image.norm.vmax), (-0.4, 0.4))
            self.assertEqual(image.cmap(image.norm(0)), to_rgba("white"))
            np.testing.assert_array_equal(right.get_yticks(), range(9))
            self.assertIn("0.01", right.get_legend().texts[0].get_text())
        finally:
            plt.close(figure)

    def test_candidate_mean_equal_weights_single_candidate_and_gap(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        def group(values):
            result = {k: {"fid": 2.1} for k in range(9)}
            result.update({k: {"fid": value} for k, value in values.items()})
            return result

        groups = {("20k", 50000): group({1: 2.0, 2: 2.01, 4: 2.02, 8: 2.0200001}),
                  ("40k", 50000): {0: {"fid": 2.0}},
                  ("60k", 50000): group({7: 2.0})}
        figure = plotting.make_dots_figure(groups, 50000)
        try:
            line = figure.axes[1].lines[0]
            np.testing.assert_allclose(line.get_ydata()[:3], [7 / 3, np.nan, 7], equal_nan=True)
            np.testing.assert_array_equal(line.get_xdata(), range(10))
            self.assertEqual(line.get_color(), "#ff7f0e")
            self.assertEqual(line.get_marker(), "o")
            self.assertEqual(line.get_label(), "Mean K (FID gap <= 0.02)")
            # A NaN vertex must split the line rather than bridge the missing checkpoint.
            from matplotlib.path import Path
            moves = [code for _, code in line.get_path().iter_segments() if code == Path.MOVETO]
            self.assertEqual(len(moves), 2)
        finally:
            plt.close(figure)

    def test_distances_ties_missing_columns_and_overflow(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, MPLCONFIGDIR=temporary):
                figure, axis = plt.subplots()
                try:
                    complete = {k: {"fid": 2.0 if k in (2, 3) else 2.1} for k in range(9)}
                    complete[1]["fid"] = 2.01
                    groups = {("40k", 50000): complete,
                              ("60k", 50000): {0: {"fid": 1.0}}}
                    plotting.draw_near_optimal(axis, groups, 50000)
                    image = axis.images[0]
                    matrix = image.get_array()
                    np.testing.assert_allclose(matrix[:, 1],
                                               [0.1, 0.01, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1])
                    self.assertTrue(np.ma.getmaskarray(matrix)[:, [0, 2]].all())
                    np.testing.assert_allclose(axis.collections[0].get_offsets(), [[1, 2]])
                    self.assertEqual(len(axis.lines), 0)
                    self.assertEqual(image.origin, "lower")
                    self.assertEqual((image.norm.vmin, image.norm.vmax), (0, 0.05))
                    self.assertEqual(image.cmap(image.norm(0)), to_rgba("#08306b"))
                    self.assertEqual(image.cmap(image.norm(0.05)), to_rgba("white"))
                    self.assertEqual(image.cmap(image.norm(0.1)), to_rgba("white"))
                    self.assertEqual(image.cmap.get_bad().tolist(), list(to_rgba("#eeeeee")))
                    self.assertEqual(image.colorbar.extend, "max")
                    np.testing.assert_allclose(image.colorbar.get_ticks(), [0, 0.01, 0.02, 0.05])
                    axis.clear()
                    plotting.draw_near_optimal(axis, {("20k", 5000):
                        {k: {"fid": 2.0} for k in range(9)}}, 5000)
                    self.assertEqual(axis.images[0].colorbar.extend, "neither")
                    np.testing.assert_allclose(axis.collections[0].get_offsets(), [[0, 0]])
                finally:
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
