#!/usr/bin/env python3
"""CPU-only integration tests for the shared runner's condition-boundary pause."""

import csv
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "experiments/8/1_restart_sampling_checkpoint_sweep/scripts/run_restart_checkpoint_sweep.sh"

FAKE_GPU = '''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
root = Path(os.environ["TEST_ROOT"])
with (root / "events").open("a") as handle:
    handle.write("gpu " + " ".join(sys.argv[1:]) + "\\n")
if "--query-gpu=name" in sys.argv:
    print("Mock GPU")
else:
    trigger = root / "repause-gpu"
    if trigger.exists():
        trigger.unlink()
        (root / "PAUSE").touch()
    if (root / "gpu-busy").exists():
        print("1234, mock-job, 1")
'''

FAKE_TORCHRUN = '''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
root = Path(os.environ["TEST_ROOT"])
phase = "generate" if "generate.py" in sys.argv else "fid"
def event(message):
    with (root / "events").open("a") as handle:
        handle.write(message + "\\n")
def option(name):
    return next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith(name + "="))
folder = Path(option("--outdir" if phase == "generate" else "--images"))
condition = folder.parent.name
event(phase + " start " + condition + " gpu=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
trigger = root / ("pause-" + phase + "-" + condition)
if trigger.exists():
    trigger.unlink()
    (root / "PAUSE").touch()
if (root / ("fail-" + phase)).exists():
    sys.exit(3)
if phase == "generate":
    folder = folder / "000000"
    folder.mkdir(parents=True, exist_ok=True)
    for seed in range(2):
        (folder / (f"{seed:06d}.png")).write_bytes(b"mock PNG")
else:
    print("2.5")
event(phase + " done " + condition)
'''


class ConditionPauseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="edm-condition-pause-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.exp = self.root / "experiment"
        self.exp.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, contents in [("nvidia-smi", FAKE_GPU), ("torchrun", FAKE_TORCHRUN)]:
            path = self.bin / name
            path.write_text(contents)
            path.chmod(0o755)
        for name in ["network.pkl", "ref.npz"]:
            (self.root / name).touch()
        self.config = self.root / "config.tsv"
        self.set_conditions("a", "b")
        self.gpu_config = self.root / "gpu.env"
        self.gpu_config.write_text('GPU_INDEX=7\nEXPECTED_GPU_NAME="Mock GPU"\n')
        self.env = dict(
            os.environ, TEST_ROOT=str(self.root), PATH=f"{self.bin}:{os.environ['PATH']}",
            EXP_ROOT=str(self.exp), CONFIG=str(self.config),
            TORCHRUN=str(self.bin / "torchrun"), FID_REF=str(self.root / "ref.npz"),
            NUM_IMAGES="2", FID_SEEDS="0-1", FID_GENERATION_SEEDS="0-1",
            SAMPLE_SEEDS="-", SAMPLES_ONLY="0", PLAN_ONLY="0",
            CONDITION_FILTER="", CHECKPOINT_FILTER="", PAUSE_ENABLED="1",
            PAUSE_FILE=str(self.root / "PAUSE"), PAUSE_POLL_SEC="0.03",
            PAUSE_ETA_UPDATE_SEC="0.10",
            EDM_GPU_CONFIG=str(self.gpu_config), CHECK_INTERVAL_SEC="1",
            REQUIRE_EXACT_IMAGE_COUNT="1", REQUIRE_EXACT_SEED_COVERAGE="1",
            GPU_INDEX="99", EXPECTED_GPU_NAME="Wrong GPU", CUDA_VISIBLE_DEVICES="99",
            IMAGE_ROOT=str(self.exp / "images"), RESULTS_DIR=str(self.exp / "results"),
            LOG_DIR=str(self.exp / "logs"), MASTER_LOG=str(self.exp / "logs/master.log"),
        )
        self.process = None

    def set_conditions(self, *names):
        self.config.write_text("".join(
            f"{name}\t20k\t18\t0\t0\t0\t0\t35\ttest\t{self.root / 'network.pkl'}\n"
            for name in names
        ))

    def start(self):
        self.output = (self.root / "output").open("w")
        self.addCleanup(self.output.close)
        self.process = subprocess.Popen(
            ["bash", str(RUNNER)], cwd=REPO_ROOT, env=self.env,
            stdout=self.output, stderr=subprocess.STDOUT,
        )
        self.addCleanup(self.stop_process)

    def stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def wait_for(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if self.process.poll() is not None:
                break
            time.sleep(0.02)
        self.fail((self.root / "output").read_text())

    def text(self, name="events"):
        path = self.root / name
        return path.read_text() if path.exists() else ""

    def pause_log(self):
        path = self.exp / "logs/pause.log"
        return path.read_text() if path.exists() else ""

    def rows(self):
        path = self.exp / "results/restart_fid_summary.tsv"
        if not path.exists():
            return []
        with path.open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def finish(self, code=0):
        self.assertEqual(self.process.wait(timeout=8), code, self.text("output"))

    def test_pause_during_generation_saves_fid_then_waits_and_resumes(self):
        (self.root / "pause-generate-a").touch()
        self.start()
        self.wait_for(lambda: "PAUSE before [b FID-2]" in self.pause_log())
        self.assertEqual([row["condition"] for row in self.rows()], ["a"])
        self.assertIn("fid done a", self.text())
        self.assertNotIn("generate start b", self.text())
        (self.root / "PAUSE").unlink()
        self.finish()
        self.assertEqual(len(self.rows()), 2)
        self.assertIn("generate start b gpu=7", self.text())
        self.assertIn("gpu -i 7 --query-gpu=name", self.text())
        self.assertIn("RESUME GPU free", self.pause_log())

    def test_repeated_pauses_during_fid_save_before_waiting(self):
        self.set_conditions("a", "b", "c")
        for condition in ["a", "b"]:
            (self.root / f"pause-fid-{condition}").touch()
        self.start()
        for count, next_condition in [(1, "b"), (2, "c")]:
            self.wait_for(lambda: f"PAUSE before [{next_condition} FID-2]" in self.pause_log())
            self.assertEqual(len(self.rows()), count)
            self.assertNotIn(f"generate start {next_condition}", self.text())
            (self.root / "PAUSE").unlink()
        self.finish()
        self.assertEqual(len(self.rows()), 3)

    def test_initial_pause_rechecks_pause_after_gpu_wait(self):
        (self.root / "PAUSE").touch()
        (self.root / "repause-gpu").touch()
        self.start()
        self.wait_for(lambda: "PAUSE before [a FID-2]" in self.pause_log())
        self.assertEqual(self.text(), "")
        (self.root / "PAUSE").unlink()
        self.wait_for(lambda: self.pause_log().count("PAUSE before [a FID-2]") == 2)
        self.assertNotIn("generate start", self.text())
        (self.root / "PAUSE").unlink()
        self.finish()

    def test_resume_waits_for_busy_gpu(self):
        (self.root / "PAUSE").touch()
        (self.root / "gpu-busy").touch()
        self.start()
        self.wait_for(lambda: "PAUSE before" in self.pause_log())
        (self.root / "PAUSE").unlink()
        self.wait_for(lambda: "still busy" in self.text("output"))
        self.assertNotIn("generate start", self.text())
        (self.root / "gpu-busy").unlink()
        self.finish()

    def test_last_condition_finishes_without_waiting(self):
        self.set_conditions("a")
        (self.root / "pause-fid-a").touch()
        self.start()
        self.finish()
        self.assertEqual(len(self.rows()), 1)
        self.assertTrue((self.root / "PAUSE").exists())
        self.assertEqual(self.pause_log(), "")

    def test_transient_pause_removed_before_boundary_does_not_wait(self):
        (self.root / "pause-generate-a").touch()
        fake = self.bin / "torchrun"
        fake.write_text(FAKE_TORCHRUN.replace(
            'print("2.5")',
            'print("2.5"); (root / "PAUSE").unlink(missing_ok=True)',
        ))
        self.start()
        self.finish()
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.pause_log(), "")

    def test_skips_completed_next_condition_without_waiting(self):
        self.start()
        self.finish()
        self.process = None
        # Retain b's result; rerun only the now-unrecorded a before b.
        summary = self.exp / "results/restart_fid_summary.tsv"
        lines = summary.read_text().splitlines(keepends=True)
        summary.write_text(lines[0] + lines[2])
        (self.root / "pause-fid-a").touch()
        self.start()
        self.finish()
        self.assertEqual({row["condition"] for row in self.rows()}, {"a", "b"})
        self.assertEqual(self.pause_log(), "")

    def test_completed_conditions_do_not_wait_or_require_gpu_config(self):
        self.start()
        self.finish()
        self.process = None
        (self.root / "PAUSE").touch()
        self.gpu_config.unlink()
        self.start()
        self.finish()
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.pause_log(), "")

    def test_exit_stop_takes_priority_while_paused(self):
        (self.root / "PAUSE").touch()
        self.start()
        self.wait_for(lambda: "PAUSE before" in self.pause_log())
        (self.exp / "STOP_AFTER_CURRENT_CONDITION").touch()
        self.finish()
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.text(), "")

    def test_exit_stop_after_condition_takes_priority_over_pause(self):
        (self.root / "pause-fid-a").touch()
        # Have the mock FID create both requests during the first condition.
        fake = self.bin / "torchrun"
        fake.write_text(FAKE_TORCHRUN.replace(
            '(root / "PAUSE").touch()',
            '(root / "PAUSE").touch(); (Path(os.environ["EXP_ROOT"]) / "STOP_AFTER_CURRENT_CONDITION").touch()',
        ))
        self.start()
        self.finish()
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.pause_log(), "")

    def test_plan_only_ignores_pause_and_missing_gpu_config(self):
        self.env["PLAN_ONLY"] = "1"
        self.gpu_config.unlink()
        (self.root / "PAUSE").touch()
        self.start()
        self.finish()
        self.assertIn("PLAN [a checkpoint=20k", self.text("output"))
        self.assertEqual(self.text(), "")

    def test_other_experiment_default_is_unchanged(self):
        self.env.pop("PAUSE_ENABLED")
        self.gpu_config.unlink()
        (self.root / "PAUSE").touch()
        self.start()
        self.finish()
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.pause_log(), "")

    def test_8_4_default_enabled_for_direct_shared_runner(self):
        self.env.pop("PAUSE_ENABLED")
        self.env["EXP_ROOT"] = str(REPO_ROOT / "experiments/8/4_restart_k_sweep_10k_interval")
        self.env["PLAN_ONLY"] = "1"
        self.start()
        self.finish()
        self.assertIn("Condition-boundary pause:", self.text("output"))

    def test_child_failure_is_not_retried(self):
        (self.root / "fail-fid").touch()
        (self.root / "pause-fid-a").touch()
        self.start()
        self.finish(code=3)
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.pause_log(), "")
        self.assertEqual(self.text().count("fid start a"), 1)
        self.assertIn("エラー終了", (self.root / "PAUSE").read_text())

    def test_missing_gpu_policy_blocks_pending_work(self):
        self.gpu_config.unlink()
        self.start()
        self.finish(code=1)
        self.assertIn("Missing local GPU configuration", self.text("output"))
        self.assertEqual(self.text(), "")

    def test_eta_updates_during_generation_then_records_waiting(self):
        (self.root / "pause-generate-a").touch()
        hold = self.root / "hold"
        hold.touch()
        fake = self.bin / "torchrun"
        fake.write_text(FAKE_TORCHRUN.replace(
            'if (root / ("fail-" + phase)).exists():',
            'if phase == "generate" and condition == "a":\n'
            '    import time\n'
            '    print("10/20 [02:00<02:00, 12s/batch]", flush=True)\n'
            '    while (root / "hold").exists(): time.sleep(0.02)\n'
            'if (root / ("fail-" + phase)).exists():',
        ))
        self.start()
        pause = self.root / "PAUSE"
        self.wait_for(lambda: pause.exists() and pause.read_text().count("状態=画像生成中") >= 2)
        self.assertEqual(len(self.rows()), 0)
        self.assertNotIn("fid start a", self.text())
        hold.unlink()
        self.wait_for(lambda: "状態=待機中" in pause.read_text())
        self.assertEqual(len(self.rows()), 1)
        self.assertIn("待機開始まで=0秒", pause.read_text())
        pause.unlink()
        self.finish()
        self.assertFalse(pause.exists())
        state = self.exp / f"logs/pause-runtime.{self.process.pid}.json"
        import json
        self.assertEqual(json.loads(state.read_text())["phase"], "complete")

    def test_stop_retains_complete_message_after_final_condition(self):
        self.set_conditions("a")
        (self.root / "pause-fid-a").touch()
        self.start()
        self.finish()
        self.assertIn("実験終了・待機不要", (self.root / "PAUSE").read_text())


if __name__ == "__main__":
    unittest.main()
