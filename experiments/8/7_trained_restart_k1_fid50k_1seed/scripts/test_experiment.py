#!/usr/bin/env python3
"""CPU checks for Experiment 8-7 NFE, paired FID, image coverage, and resume."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_experiment as runner
from summarize_results import summarize

sys.path.insert(0, str(runner.REPO_ROOT))
import torch
import generate


class CountingDenoiser:
    sigma_min = 0.002
    sigma_max = 80

    def __init__(self):
        self.calls = 0

    def __call__(self, x, sigma, class_labels=None):
        self.calls += 1
        return torch.zeros_like(x)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def result(method: str, fid: float) -> dict:
    return dict(checkpoint="exp3-best", num_images="50000", seeds="0-49999",
                nfe="39", fid_ref="reference.npz", method=method,
                main_steps="18" if method == "restart" else "20",
                restart_steps="3" if method == "restart" else "0",
                restart_k="1" if method == "restart" else "0", fid=str(fid),
                network="checkpoint.pkl", rho="7", restart_noise="1", gen_batch="64", fid_batch="64")


class TestExperiment(unittest.TestCase):
    def test_actual_network_calls_are_39_for_both_conditions(self):
        latents = torch.randn([2, 3, 4, 4], generator=torch.Generator().manual_seed(0))
        restart, ode = CountingDenoiser(), CountingDenoiser()
        info = '{"0":[3,1,0.06,0.30]}'
        generate.restart_sampler(restart, latents, num_steps=18, restart_info=info,
                                 randn_like=lambda x: torch.zeros_like(x))
        generate.edm_sampler(ode, latents, num_steps=20, randn_like=lambda x: torch.zeros_like(x))
        self.assertEqual(generate.restart_nfe(18, info), 39)
        self.assertEqual((restart.calls, ode.calls), (39, 39))

    def test_improvement_sign_and_incomplete_pair(self):
        for restart, ode, delta in ((1.7, 1.9, 0.2), (2.1, 1.9, -0.2)):
            pair = summarize([result("restart", restart), result("ode-heun", ode)])[0]
            self.assertEqual(pair["complete"], 1)
            self.assertAlmostEqual(float(pair["improvement_vs_same_nfe"]), delta)
        partial = summarize([result("restart", 1.7)])[0]
        self.assertEqual((partial["complete"], partial["improvement_vs_same_nfe"]), (0, "NA"))

    def test_reject_duplicate_and_mismatched_protocol(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([result("restart", 1.7), result("restart", 1.8)])
        ode = result("ode-heun", 1.9)
        ode["network"] = "other.pkl"
        with self.assertRaisesRegex(ValueError, "Mismatched network"):
            summarize([result("restart", 1.7), ode])

    def test_exact_seed_coverage_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "images"
            runner.ensure_protocol(directory, {"seeds": "0-1"})
            self.assertFalse(runner.validate_images(directory, [0, 1], subdirs=True))
            (directory / "000000").mkdir()
            for seed in (0, 1):
                (directory / "000000" / f"{seed:06d}.png").touch()
            self.assertTrue(runner.validate_images(directory, [0, 1], subdirs=True))
            (directory / "000000.png").touch()
            with self.assertRaisesRegex(ValueError, "Unexpected images"):
                runner.validate_images(directory, [0, 1], subdirs=True)
            with self.assertRaisesRegex(ValueError, "Protocol changed"):
                runner.ensure_protocol(directory, {"seeds": "0-2"})

    def test_plan_is_read_only_and_resume_skips_completed_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, ref = root / "network.pkl", root / "reference.npz"
            checkpoint.touch()
            ref.touch()
            config = root / "config.tsv"
            rows = runner.load_conditions(runner.DEFAULT_ROOT / "config" / "restart_k1_fid50k.tsv")
            config_fields = list(rows[0])[:-1]
            with config.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=config_fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row, network=str(checkpoint)))
            output = root / "output"
            environment = {"EXP_ROOT": str(output), "CONFIG": str(config), "FID_REF": str(ref),
                           "PLAN_ONLY": "1", "SAMPLES_ONLY": "0", "SAMPLE_SEEDS": "-",
                           "CHECKPOINT_FILTER": "", "CONDITION_FILTER": "", "TORCHRUN": sys.executable,
                           "GEN_BATCH": "64", "FID_BATCH": "64"}
            with patch.dict(os.environ, environment), patch.object(sys, "argv", ["run_experiment.py"]):
                runner.main()
                self.assertFalse(output.exists())
                os.environ["PLAN_ONLY"] = "0"
                # Test orchestration without running GPU children or producing fake image artifacts.
                with patch.object(runner, "validate_images", return_value=True), \
                     patch.object(runner, "run_logged", return_value=(1.0, "1.9")) as launch, \
                     patch.object(runner, "aggregate"):
                    runner.main()
                    self.assertEqual(launch.call_count, 4)
                    recorded = runner.read_results(output / "results" / "restart_fid_summary.tsv")
                    self.assertEqual(len(recorded), 4)
                    self.assertEqual(sum(row["complete"] for row in summarize(recorded)), 2)
                    launch.reset_mock()
                    runner.main()
                    launch.assert_not_called()
                    os.environ["GEN_BATCH"] = "32"
                    with self.assertRaisesRegex(ValueError, "Recorded protocol changed"):
                        runner.main()
                (output / "STOP_AFTER_CURRENT_CONDITION").touch()
                with patch.object(runner, "run_logged") as launch:
                    runner.main()
                    launch.assert_not_called()

    def test_cli_aggregation_and_plot_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, improvements = root / "summary.tsv", root / "improvements.tsv"
            fixtures = [result("restart", 1.7), result("ode-heun", 1.9),
                        dict(result("restart", 2.1), checkpoint="pretrained"),
                        dict(result("ode-heun", 1.9), checkpoint="pretrained")]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fixtures[0]), delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(fixtures)
            subprocess.run([sys.executable, str(runner.SCRIPT_ROOT / "summarize_results.py"),
                            "--input", str(source), "--output", str(improvements)], check=True)
            subprocess.run([sys.executable, str(runner.SCRIPT_ROOT / "plot_results.py"),
                            "--input", str(improvements), "--output", str(root / "overview")],
                           env=dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib")), check=True)
            with improvements.open(newline="", encoding="utf-8") as handle:
                comparisons = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([float(row["improvement_vs_same_nfe"]) for row in comparisons], [0.2, -0.2])
            self.assertEqual((root / "overview.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn("Experiment 8-7", (root / "overview.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
