#!/usr/bin/env python3
"""Build quality-controlled observations and the leakage-audited model table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pm25ml.data import prepare_all  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(prepare_all(), indent=2, default=str))
