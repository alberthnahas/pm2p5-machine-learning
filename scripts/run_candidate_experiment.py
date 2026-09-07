#!/usr/bin/env python3
"""Run the predeclared availability-aligned candidate development experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.candidate import run_candidate_experiment  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_candidate_experiment(), indent=2, default=str))
