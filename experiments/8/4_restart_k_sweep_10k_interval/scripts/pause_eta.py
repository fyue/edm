#!/usr/bin/env python3
"""Compatibility entrypoint for the shared pause ETA helper."""
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("_shared_pause_eta", Path(__file__).resolve().parents[4] / "scripts/pause_eta.py")
shared = importlib.util.module_from_spec(spec)
sys.modules[__name__] = shared
spec.loader.exec_module(shared)
if __name__ == "__main__":
    shared.main()
