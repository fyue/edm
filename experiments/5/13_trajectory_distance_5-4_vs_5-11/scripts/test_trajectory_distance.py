#!/usr/bin/env python3
"""CPU checks for the Experiment 5-13 trajectory implementations."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate import ab_sampler, pndm_sampler
from trajectory_distance import (
    build_sigma_steps,
    compute_batch_rows,
    estimate_nfe,
    plot_stats,
    run_ab_trajectory,
    summarize_rows,
    write_stats,
)


class FakeDenoiser(torch.nn.Module):
    img_channels = 1
    img_resolution = 2
    label_dim = 0
    sigma_min = 0.0
    sigma_max = float("inf")

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del class_labels
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        return 0.7 * x + 0.01 * sigma


class TrajectoryDistanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.net = FakeDenoiser()
        generator = torch.Generator().manual_seed(123)
        self.latents = torch.randn(2, 1, 2, 2, generator=generator)
        self.sigma_steps = build_sigma_steps(
            self.net,
            steps=6,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            device=torch.device("cpu"),
        )
        self.x_initial = self.latents.to(torch.float64) * self.sigma_steps[0]

    def test_variable_ab_matches_generate_sampler(self) -> None:
        observed = run_ab_trajectory(
            self.net,
            self.x_initial,
            None,
            self.sigma_steps,
            variable=True,
        )["final"]
        expected = ab_sampler(self.net, self.latents, num_steps=6)
        self.assertTrue(torch.equal(observed, expected))

    def test_fixed_ab_matches_generate_sampler(self) -> None:
        observed = run_ab_trajectory(
            self.net,
            self.x_initial,
            None,
            self.sigma_steps,
            variable=False,
        )["final"]
        expected = pndm_sampler(
            self.net,
            self.latents,
            num_steps=6,
            rk_warmup=False,
        )
        torch.testing.assert_close(observed, expected, rtol=0, atol=2e-14)

    def test_metric_row_count_and_nfe(self) -> None:
        rows = compute_batch_rows(
            self.net,
            seeds=[0, 1],
            sigma_steps=self.sigma_steps,
            subdivisions=(1, 2),
            epsilon=1e-12,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(rows), 2 * (6 * 7 + 3))
        self.assertEqual(estimate_nfe(18, (4, 8)), 854)

        stats = summarize_rows(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "step_stats.tsv"
            plot_path = Path(tmpdir) / "trajectory_distances.svg"
            write_stats(stats_path, stats)
            plot_stats(plot_path, stats)
            self.assertGreater(stats_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
