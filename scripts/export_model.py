#!/usr/bin/env python3
"""Deprecated alias — prefer scripts/train_export.py."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    print("Note: scripts/export_model.py is a deprecated alias; prefer train_export.py")
    runpy.run_path(str(Path(__file__).with_name("train_export.py")), run_name="__main__")
