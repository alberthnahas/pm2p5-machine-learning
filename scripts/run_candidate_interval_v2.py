#!/usr/bin/env python3
"""Run the one predeclared input-scaled candidate interval correction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.candidate import run_interval_scale_correction  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_interval_scale_correction(), indent=2, default=str))
