#!/usr/bin/env python3
"""Generate all scientific figures in vector and high-resolution raster form."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.plots import make_all_figures


if __name__ == "__main__":
    print(json.dumps(make_all_figures(), indent=2))
