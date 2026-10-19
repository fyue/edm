"""CPU tests for nested seed reuse, resume safety and comparison outputs."""

import argparse
import contextlib
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment88", SCRIPT_ROOT / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="edm-exp88-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        network, ref = self.root / "network.pkl", self.root / "ref.npz"
        network.write_bytes(b"test checkpoint")
        ref.write_bytes(b"test reference")
        self.row = dict(condition="20k-tmin014", checkpoint="20k", main_steps=18,
                        restart_steps=3, restart_k=5, t_min=0.14, t_max=0.30,
                        nfe=55, network=str(network))
        self.args = argparse.Namespace(exp_root=self.root / "run", fid_ref=ref,
                                       gen_batch=64, fid_batch=64, torchrun="torchrun")

    def png(self, directory, seed):
        path = directory / experiment.image_relative(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (seed % 255, 3, 4)).save(path)
        return path

    def test_inventory_and_hardlink_integrity(self):
        source, pool = self.root / "source", self.root / "pool"
        first = self.png(source, 0)
        self.png(source, 1)
        original = first.read_bytes()
        experiment.inventory(source, range(2), complete=True)
        experiment.link_image(first, pool / experiment.image_relative(0))
        self.assertTrue(os.path.samefile(first, pool / experiment.image_relative(0)))
        self.assertEqual(first.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "missing seeds"):
            experiment.inventory(pool, range(2), complete=True)
        self.png(pool, 2)
        with self.assertRaisesRegex(ValueError, "Unexpected seed"):
            experiment.inventory(pool, range(2))
        with mock.patch.object(experiment.os, "link", side_effect=OSError(errno.EXDEV, "different filesystem")):
            target = self.root / "copy.png"
            experiment.link_image(first, target)
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(os.path.samefile(first, target))

    def test_protocol_guard(self):
        value = experiment.protocol(self.row, self.args)
        directory = self.root / "pool"
        experiment.ensure_protocol(directory, value, create=False)
        self.assertFalse(directory.exists())
        experiment.ensure_protocol(directory, value)
        experiment.ensure_protocol(directory, value)
        changed = dict(value, gen_batch=32)
        with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
            experiment.ensure_protocol(directory, changed)

    def test_corrupt_png_is_rejected(self):
        directory = self.root / "corrupt"
        path = self.png(directory, 0)
        path.write_bytes(b"broken PNG")
        with self.assertRaises(OSError):
            experiment.inventory(directory, range(1), complete=True)

    def test_synced_protocol_preserves_id_and_rejects_file_changes(self):
        for name in ("generate.py", "fid.py"):
            (self.root / name).write_text("# fixture\n")
        with mock.patch.object(experiment, "REPO_ROOT", self.root):
            value = experiment.protocol(self.row, self.args)
            recorded = json.loads(json.dumps(value))
            old_root = Path("/other-pc/projects/edm")
            for key in ("network", "fid_ref", "generate", "fid"):
                recorded[key]["path"] = str(old_root / Path(value[key]["path"]).relative_to(self.root))
            recorded["condition"]["network"] = str(old_root / "network.pkl")
            pool = self.args.exp_root / "images" / self.row["condition"] / "pool"
            experiment.ensure_protocol(pool, recorded)
            original = (pool / ".protocol.json").read_bytes()
            resumed = experiment.protocol(self.row, self.args)
            self.assertEqual(resumed, recorded)
            experiment.ensure_protocol(pool, resumed)
            self.assertEqual((pool / ".protocol.json").read_bytes(), original)
            self.args.gen_batch = 32
            with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
                experiment.ensure_protocol(pool, experiment.protocol(self.row, self.args))
            self.args.gen_batch = 64
            (self.root / "network.pkl").write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
                experiment.ensure_protocol(pool, experiment.protocol(self.row, self.args))

    def test_nested_5k_25k_50k_and_resume_generate_only_new_seeds(self):
        pool = self.args.exp_root / "images" / self.row["condition"] / "pool"
        (self.args.exp_root / "logs").mkdir(parents=True)
        present, commands = set(), []

        def inventory(directory, allowed, complete=False, verify=True):
            seeds = present & set(allowed)
            if complete:
                self.assertEqual(seeds, set(allowed))
            return {seed: pool / experiment.image_relative(seed) for seed in seeds}

        def run(command, log):
            commands.append(command)
            if "generate.py" in command:
                seeds = next(value.split("=", 1)[1] for value in command if value.startswith("--seeds="))
                new = set(experiment.parse_seeds(seeds))
                self.assertFalse(present & new, "must never regenerate existing seeds")
                present.update(new)
                return 2.0, None
            return 1.0, 7.0

        with mock.patch.object(experiment, "inventory", side_effect=inventory), \
             mock.patch.object(experiment, "link_image"), \
             mock.patch.object(experiment, "run_logged", side_effect=run), \
             mock.patch.object(experiment, "aggregate"):
            results = []
            experiment.evaluate(self.row, 5000, self.args, results, None)
            experiment.evaluate(self.row, 25000, self.args, results, None)
            base_bytes = (pool / ".protocol.json").read_bytes()
            original_results = {path: path.read_bytes() for path in (self.args.exp_root / "results").glob("*-fid*.json")}
            experiment.evaluate(self.row, 50000, self.args, results, None)
            self.assertEqual((pool / ".protocol.json").read_bytes(), base_bytes)
            for path, contents in original_results.items():
                self.assertEqual(path.read_bytes(), contents)
            before = len(commands)
            experiment.evaluate(self.row, 5000, self.args, results, None)
            experiment.evaluate(self.row, 25000, self.args, results, None)
            experiment.evaluate(self.row, 50000, self.args, results, None)
            self.assertEqual(len(commands), before)
        seed_options = [value for command in commands if "generate.py" in command
                        for value in command if value.startswith("--seeds=")]
        self.assertEqual(seed_options, ["--seeds=0-4999", "--seeds=5000-24999", "--seeds=25000-49999"])
        self.assertEqual(present, set(range(50000)))
        self.assertEqual(len(experiment.load_results(self.args.exp_root)), 3)
        extension = json.loads((pool / ".extension.json").read_text())
        self.assertEqual(extension["additional_seeds"], "25000-49999")
        self.assertEqual(extension["base_protocol_id"], experiment.digest(experiment.protocol(self.row, self.args)))

    def test_missing_seed_ranges(self):
        self.assertEqual(experiment.seed_ranges([0, 1, 4, 7, 8]), ["0-1", "4-4", "7-8"])
        self.assertEqual(experiment.seed_ranges([]), [])

    def test_comparison_sign_partial_and_plots(self):
        results = []
        for count in experiment.STAGES:
            for checkpoint in experiment.CHECKPOINTS:
                for t_min, fid in ((0.06, 7.0), (0.14, 6.5)):
                    results.append(dict(condition=f"{checkpoint}-test-{t_min}", checkpoint=checkpoint, num_images=count,
                                        seeds=f"0-{count - 1}", t_min=t_min, fid=fid))
        comparisons = experiment.comparison_rows(results)
        self.assertEqual([row["delta"] for row in comparisons], [0.5] * 12)
        self.assertEqual([row["checkpoint"] for row in comparisons[::3]], list(experiment.CHECKPOINTS))
        self.assertEqual(experiment.comparison_rows(results[:1])[0]["delta"], "NA")
        # Aggregation integration test uses the actual output schema, only synthetic FIDs.
        full = []
        for row in results:
            values = {field: "" for field in experiment.FIELDS}
            values.update(row)
            full.append(values)
        with mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(self.root / "mpl")}):
            experiment.aggregate(self.root / "output", full)
        for count in experiment.STAGES:
            for extension in ("png", "svg"):
                self.assertTrue((self.root / f"output/results/restart_tmin_fid{count}.{extension}").is_file())
                self.assertTrue((self.root / f"output/results/restart_tmin_fid{count}_lines.{extension}").is_file())
            svg = (self.root / f"output/results/restart_tmin_fid{count}_lines.svg").read_text()
            self.assertIn("170k best (Exp.3-10)", svg)
            self.assertIn("not a training trajectory", svg)
        self.assertEqual(json.loads((self.root / "output/results/tmin_comparison.json").read_text())[0]["delta"], 0.5)

    def test_plan_only_is_read_only_and_fixed_config(self):
        rows = experiment.load_conditions(experiment.DEFAULT_ROOT / "config/restart_tmin.tsv")
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["restart_k"] for row in rows}, {5})
        self.assertEqual({row["nfe"] for row in rows}, {55})
        outdir = self.root / "no-output"
        completed = subprocess.run([sys.executable, str(SCRIPT_ROOT / "run_experiment.py"),
                                    "--plan-only", "--fid-stage", "all", "--exp-root", str(outdir)],
                                   capture_output=True, text=True, check=True)
        self.assertFalse(outdir.exists())
        self.assertEqual(completed.stdout.count("main=18 restart_steps=3 K=5 NFE=55"), 8)
        self.assertIn("--seeds=5000-24999", completed.stdout)
        self.assertIn("--seeds=25000-49999", completed.stdout)

    def test_extension_guard_rejects_changed_id(self):
        value = experiment.protocol(self.row, self.args)
        pool = self.root / "pool"
        with self.assertRaisesRegex(ValueError, "Missing 50k"):
            experiment.ensure_extension(pool, value, create=False)
        experiment.ensure_extension(pool, value)
        experiment.ensure_extension(pool, value, create=False)
        with self.assertRaisesRegex(ValueError, "Extension metadata mismatch"):
            experiment.ensure_extension(pool, dict(value, restart_noise=2))

    def test_legacy_reuse_salvages_valid_seeds_and_preserves_existing(self):
        source_root = self.root / "legacy"
        directory = source_root / "images/20k-restart-k5/pool"
        row = dict(self.row, t_min=0.06)
        for seed in (0, 1, 2):
            self.png(directory, seed)
        (directory / experiment.image_relative(1)).write_bytes(b"broken PNG")
        source = dict(row, condition="20k-restart-k5", method="restart", num_images=5000,
                      seeds="0-4999", images=str(directory))
        experiment.write_tsv(source_root / "results/restart_fid_summary.tsv", [source], list(source))
        config = source_root / "config/restart_k5_fid50k.tsv"
        config.parent.mkdir()
        config.write_text("\t".join(["20k-restart-k5", "20k", "18", "3", "5", "0.06", "0.30", "55", "test", row["network"]]) + "\n")
        candidate, _ = experiment.source_candidate(row, source_root, 64)
        self.assertEqual(candidate["_valid_seeds"], [0, 2])
        pool = self.args.exp_root / "images" / row["condition"] / "pool"
        value = experiment.protocol(row, self.args)
        experiment.ensure_protocol(pool, value)
        own = self.png(pool, 0)
        original = own.read_bytes()
        experiment.prepare_pool(row, self.args, value, [candidate])
        self.assertEqual(own.read_bytes(), original)
        self.assertTrue(os.path.samefile(directory / experiment.image_relative(2), pool / experiment.image_relative(2)))
        self.assertFalse((pool / experiment.image_relative(1)).exists())
        self.assertEqual(experiment.reused_provenance(pool, 5000)[0], 1)

    def test_existing_result_protocol_ids_remain_compatible(self):
        args = argparse.Namespace(exp_root=experiment.DEFAULT_ROOT, gen_batch=64, fid_batch=64,
                                  fid_ref=experiment.REPO_ROOT / "fid-refs/cifar10-32x32.npz")
        rows = {row["condition"]: row for row in experiment.load_conditions(experiment.DEFAULT_ROOT / "config/restart_tmin.tsv")}
        for item in experiment.load_results(experiment.DEFAULT_ROOT):
            if item["checkpoint"] in ("20k", "40k", "60k"):
                self.assertEqual(item["protocol_id"], experiment.digest(experiment.protocol(rows[item["condition"]], args)))

    def test_completion_watch_rejects_incomplete_and_detects_run_lock(self):
        import watch_completion
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            watch_completion.verify_completion(self.root, [self.row], [])
        self.assertFalse(watch_completion.running(self.root))
        path = self.root / "logs/.run.lock"
        path.parent.mkdir()
        with path.open("w") as handle:
            watch_completion.fcntl.flock(handle, watch_completion.fcntl.LOCK_EX)
            self.assertTrue(watch_completion.running(self.root))
            watch_completion.fcntl.flock(handle, watch_completion.fcntl.LOCK_UN)
            self.assertFalse(watch_completion.running(self.root))

    def test_invalid_source_falls_back(self):
        row = dict(self.row, t_min=0.06)
        source, reason = experiment.source_candidate(row, self.root / "absent", 64)
        self.assertIsNone(source)
        self.assertIn("summary", reason)
        source, _ = experiment.source_candidate(row, experiment.SOURCE_ROOT, 32)
        self.assertIsNone(source)
        source, _ = experiment.source_candidate(row, experiment.SOURCE_ROOT, 64)
        self.assertIsNone(source, "test network differs from the official experiment checkpoint")

    def test_170k_best_rejects_other_training_branch(self):
        rows = experiment.load_conditions(experiment.DEFAULT_ROOT / "config/restart_tmin.tsv")
        row = next(row for row in rows if row["condition"] == "170k-best-tmin006")
        self.assertIn("experiments/3/10_", row["network"])
        source, reason = experiment.source_candidate(row, experiment.SOURCE_ROOT, 64)
        self.assertIsNone(source)
        self.assertIn("checkpoint file mismatch", reason)

    def test_170k_baseline_preserves_all_existing_results_and_protocols(self):
        import hashlib
        baseline = json.loads((experiment.DEFAULT_ROOT / "results/170k_extension_baseline.json").read_text())
        self.assertEqual(len(baseline["result_protocol_ids"]), 18)
        for name, checksum in baseline["file_sha256"].items():
            self.assertEqual(hashlib.sha256((experiment.DEFAULT_ROOT / name).read_bytes()).hexdigest(), checksum)

    def test_completion_wait_fails_when_runner_exits_before_lock(self):
        import watch_completion
        with mock.patch.object(sys, "argv", ["watch_completion.py", "--exp-root", str(self.root), "--wait-for-run", "--runner-pid", "12345"]), \
             mock.patch.object(watch_completion.os, "kill", side_effect=ProcessLookupError):
            self.assertEqual(watch_completion.main(), 1)
        state = json.loads((self.root / "results/completion.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["expected_evaluations"], 24)

    def test_stop_file_prevents_gpu_commands(self):
        outdir = self.root / "stopped"
        outdir.mkdir()
        (outdir / "STOP_AFTER_CURRENT_CONDITION").touch()
        with mock.patch.dict(os.environ, {"EDM_GPU_WAIT_DONE": "1"}), \
             mock.patch.object(sys, "argv", ["run_experiment.py", "--exp-root", str(outdir), "--torchrun", sys.executable]), \
             mock.patch.object(experiment, "run_logged") as run, \
             contextlib.redirect_stdout(io.StringIO()):
            experiment.main()
        run.assert_not_called()
        self.assertFalse((outdir / "images").exists())


if __name__ == "__main__":
    unittest.main()
