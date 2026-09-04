#!/usr/bin/env python3
"""Run held-out robustness and failure-mode diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.diagnostics import run_diagnostics


if __name__ == "__main__":
    print(json.dumps(run_diagnostics(), indent=2, default=str))
