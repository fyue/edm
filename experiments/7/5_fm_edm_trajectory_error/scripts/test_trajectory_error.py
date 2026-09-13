#!/usr/bin/env python3
"""CPU tests for the Experiment 7-5 trajectory analysis."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate import ablation_sampler, edm_sampler, fm_euler_sampler, fm_heun_sampler
from trajectory_error import (
    COARSE_SPECS,
    StackedRandomGenerator,
    compute_model_rows,
    endpoint_stats,
    estimate_total_nfe_per_seed,
    geometry_stats,
    path_geometry,
    plot_endpoint_errors,
    plot_path_geometry,
    run_coarse_endpoint,
    run_reference_trajectory,
)


class FakeFM(torch.nn.Module):
    img_channels = 1
    img_resolution = 2
    label_dim = 0
    sigma_min = 0.0
    sigma_max = float("inf")
    prediction_type = "velocity"

    def forward(self, x, time, class_labels=None):
        del class_labels
        time = time.to(x.dtype).reshape(-1, 1, 1, 1)
        return -0.2 * x + 0.05 * time


class FakeEDM(torch.nn.Module):
    img_channels = 1
    img_resolution = 2
    label_dim = 0
    sigma_min = 0.0
    sigma_max = float("inf")

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, class_labels=None):
        del class_labels
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        return 0.7 * x + 0.01 * sigma


class TrajectoryErrorTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.seeds = [0, 1]
        self.random = StackedRandomGenerator(self.device, self.seeds)
        self.latents = self.random.randn([2, 1, 2, 2], device=self.device)

    def test_nfe_estimate(self):
        self.assertEqual(estimate_total_nfe_per_seed(64, 128), 1846)
        self.assertEqual(sum(nfe for _, _, nfe in COARSE_SPECS), 154)

    def test_fm_coarse_endpoints_match_generate(self):
        net = FakeFM()
        observed_euler = run_coarse_endpoint(
            "fm", "euler", 5, net, self.latents, None, self.random,
        )
        observed_heun = run_coarse_endpoint(
            "fm", "heun", 3, net, self.latents, None, self.random,
        )
        torch.testing.assert_close(
            observed_euler,
            fm_euler_sampler(net, self.latents, num_steps=5),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            observed_heun,
            fm_heun_sampler(net, self.latents, num_steps=3),
            rtol=0,
            atol=0,
        )

    def test_edm_coarse_endpoints_match_generate(self):
        net = FakeEDM()
        observed_euler = run_coarse_endpoint(
            "edm", "euler", 5, net, self.latents, None, self.random,
        )
        observed_heun = run_coarse_endpoint(
            "edm", "heun", 3, net, self.latents, None, self.random,
        )
        expected_euler = ablation_sampler(
            net,
            self.latents,
            num_steps=5,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            solver="euler",
            discretization="edm",
            schedule="linear",
            scaling="none",
        )
        expected_heun = edm_sampler(net, self.latents, num_steps=3)
        torch.testing.assert_close(observed_euler, expected_euler, rtol=0, atol=0)
        torch.testing.assert_close(observed_heun, expected_heun, rtol=0, atol=0)

    def test_reference_and_metric_rows(self):
        net = FakeFM()
        initial = self.latents.to(torch.float64)
        final, states = run_reference_trajectory(
            "fm", net, initial, None, steps=8, store_states=True,
        )
        self.assertEqual(final.shape, initial.shape)
        self.assertEqual(len(states), 9)
        rows = compute_model_rows(
            "fm",
            net,
            self.seeds,
            reference_coarse_steps=4,
            reference_fine_steps=8,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            epsilon=1e-12,
            device=self.device,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(rows[1]))
        self.assertIn("euler_nfe18_relative_l2", rows[0])
        self.assertGreaterEqual(float(rows[0]["straightness_ratio"]), 1.0)

    def test_straight_path_geometry(self):
        start = torch.zeros(2, 1, 1, 1, dtype=torch.float64)
        end = torch.ones_like(start)
        states = [start, 0.25 * end, 0.5 * end, 0.75 * end, end]
        metrics = path_geometry(states, epsilon=1e-12)
        torch.testing.assert_close(
            metrics["straightness_ratio"],
            torch.ones(2, dtype=torch.float64),
        )
        torch.testing.assert_close(
            metrics["max_chord_deviation_ratio"],
            torch.zeros(2, dtype=torch.float64),
        )

    def test_aggregation_and_plots(self):
        fm_rows = compute_model_rows(
            "fm", FakeFM(), self.seeds, 2, 4, 0.002, 80, 7, 1e-12, self.device,
        )
        edm_rows = compute_model_rows(
            "edm", FakeEDM(), self.seeds, 2, 4, 0.002, 80, 7, 1e-12, self.device,
        )
        rows = [
            {key: str(value) for key, value in row.items()}
            for row in fm_rows + edm_rows
        ]
        endpoint = endpoint_stats(rows)
        geometry = geometry_stats(rows)
        self.assertEqual(len(endpoint), 20)
        self.assertEqual(len(geometry), 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            endpoint_path = Path(tmpdir) / "endpoint.svg"
            geometry_path = Path(tmpdir) / "geometry.svg"
            plot_endpoint_errors(endpoint_path, endpoint)
            plot_path_geometry(geometry_path, geometry)
            self.assertGreater(endpoint_path.stat().st_size, 0)
            self.assertGreater(geometry_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
