#!/usr/bin/env python3
"""Refit the frozen research specification into an operational model bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.deployment import build_deployment_models


if __name__ == "__main__":
    print(json.dumps(build_deployment_models(), indent=2, default=str))
