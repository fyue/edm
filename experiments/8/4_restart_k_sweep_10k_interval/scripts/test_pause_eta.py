#!/usr/bin/env python3
"""CPU tests for ETA arithmetic and safe STOP-file appending."""

import argparse
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import time
import unittest
from unittest import mock

import pause_eta


class PauseEtaTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="edm-pause-eta-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.summary = self.root / "summary.tsv"
        self.summary.write_text("num_images\tfid_ref\tfid_sec\n25000\tref.npz\t60\n")
        self.history = self.root / "history.jsonl"
        self.log = self.root / "run.log"
        self.log.write_text("old attempt: 1/2 [00:10<99:00, 1s/batch]\n")
        self.state = dict(phase="generate", started=time.monotonic(), run_id="test",
                          condition="60k-restart-k6", checkpoint="60k", k="6",
                          num_images=25000, fid_ref="ref.npz", log=str(self.log),
                          log_offset=self.log.stat().st_size)
        self.pause = self.root / "STOP"

    def progress(self, text):
        with self.log.open("a") as handle:
            handle.write(text)

    def test_generation_plus_fid_and_overhead(self):
        for phase, seconds in [("validate", 2), ("save", 1)]:
            pause_eta.append_locked(self.history, json.dumps(dict(
                host=socket.gethostname(), phase=phase, num_images=25000,
                fid_ref="ref.npz", seconds=seconds)) + "\n")
        self.progress("\r50%|x| 10/20 [02:00<02:00, 12s/batch]")
        remaining, basis = pause_eta.estimate(self.state, self.summary, self.history)
        self.assertAlmostEqual(remaining, 183, delta=1)
        self.assertIn("FID中央値", basis)

    def test_old_progress_and_absent_progress_are_not_used(self):
        remaining, _ = pause_eta.estimate(self.state, self.summary, self.history)
        self.assertIsNone(remaining)

    def test_unknown_future_fid_does_not_claim_total_eta(self):
        self.summary.unlink()
        self.progress("10/20 [02:00<02:00, 12s/batch]")
        remaining, basis = pause_eta.estimate(self.state, self.summary, self.history)
        self.assertIsNone(remaining)
        self.assertIn("FID所要時間未計測", basis)

    def test_overdue_and_finished_progress_do_not_claim_paused(self):
        self.state.update(phase="fid", started=time.monotonic() - 100)
        self.progress("20/20 [01:00<00:00, 3s/batch]\nCalculating FID...")
        remaining, basis = pause_eta.estimate(self.state, self.summary, self.history)
        self.assertIsNone(remaining)
        self.assertIn("超過", basis)
        for phase in ["waiting", "complete"]:
            self.state["phase"] = phase
            self.assertEqual(pause_eta.estimate(self.state, self.summary, self.history)[0], 0)

    def test_only_latest_five_matching_fid_results_are_used(self):
        with self.summary.open("a") as handle:
            for value in [10, 20, 30, 40, 50]:
                handle.write(f"25000\tref.npz\t{value}\n")
            handle.write("5000\tref.npz\t9999\n25000\tother.npz\t9999\n")
        self.assertEqual(pause_eta.median_fid(self.summary, self.state), 30)

    def test_existing_text_is_preserved_and_missing_stop_is_not_created(self):
        self.assertFalse(pause_eta.append_locked(self.pause, "hello", existing_only=True))
        self.assertFalse(self.pause.exists())
        self.pause.write_text("another runner's note\n")
        self.assertTrue(pause_eta.append_locked(self.pause, "our report\n", existing_only=True))
        self.assertEqual(self.pause.read_text(), "another runner's note\nour report\n")

    def test_multiple_writers_preserve_complete_records(self):
        self.pause.write_text("original note\n")

        def write(index):
            for _ in range(1000):
                if pause_eta.append_locked(self.pause, f"record-{index}\n", existing_only=True):
                    return
                time.sleep(0.001)
            raise AssertionError("Could not acquire append lock")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(40)))
        lines = self.pause.read_text().splitlines()
        self.assertEqual(lines[0], "original note")
        self.assertEqual(len(lines), 41)
        self.assertEqual(set(lines[1:]), {f"record-{i}" for i in range(40)})

    def test_replaced_stop_and_symlink_are_not_written(self):
        self.pause.touch()
        self.assertFalse(pause_eta.append_locked(self.pause, "wrong", existing_only=True, identity=(-1, -1)))
        self.assertEqual(self.pause.read_text(), "")
        self.pause.unlink()
        target = self.root / "target"
        target.write_text("preserve")
        self.pause.symlink_to(target)
        with self.assertRaises(OSError):
            pause_eta.append_locked(self.pause, "wrong", existing_only=True)
        self.assertEqual(target.read_text(), "preserve")

    def test_deletion_between_open_and_stat_does_not_recreate_stop(self):
        self.pause.touch()
        original = os.stat

        def delete_then_stat(path, **kwargs):
            self.pause.unlink()
            return original(path, **kwargs)

        with mock.patch.object(pause_eta.os, "stat", side_effect=delete_then_stat):
            self.assertFalse(pause_eta.append_locked(self.pause, "wrong", existing_only=True))
        self.assertFalse(self.pause.exists())

    def test_permission_failure_warns_without_crashing_monitor(self):
        state_file = self.root / "state.json"
        pause_eta.atomic_json(state_file, self.state)
        self.pause.touch()
        args = argparse.Namespace(state=state_file, history=self.history, summary=self.summary,
                                  pause_file=self.pause, parent_pid=os.getppid(),
                                  poll_sec=0.01, update_sec=30)
        captured = io.StringIO()
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)

        def stop_on_sleep(_seconds):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        try:
            with redirect_stdout(captured), \
                    mock.patch.object(pause_eta, "append_locked", side_effect=PermissionError("denied")), \
                    mock.patch.object(pause_eta.threading.Event, "wait", side_effect=stop_on_sleep):
                pause_eta.watch(args)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
        self.assertIn("STOP ETA warning", captured.getvalue())

    def test_report_identifies_runner_and_timezone(self):
        text = pause_eta.report(self.state, self.summary, self.history, 1234)
        self.assertIn("pid=1234", text)
        self.assertIn("checkpoint=60k K=6 FID=25000", text)
        self.assertIn("待機開始まで=推定中", text)
        self.assertRegex(text, r"T\d\d:\d\d:\d\d[+-]\d\d:\d\d")


if __name__ == "__main__":
    unittest.main()
