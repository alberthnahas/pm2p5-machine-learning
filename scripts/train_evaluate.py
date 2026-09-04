#!/usr/bin/env python3
"""Train candidate PM2.5 models and perform independent evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pm25ml.modeling import train_and_evaluate  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2, default=str))
