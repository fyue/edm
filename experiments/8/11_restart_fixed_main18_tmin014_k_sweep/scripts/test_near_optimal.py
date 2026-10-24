#!/usr/bin/env python3
"""CPU checks for near-optimal K distributions."""

import os
import tempfile
import unittest
from unittest import mock

import plot_results as plotting


class NearOptimalTests(unittest.TestCase):
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
