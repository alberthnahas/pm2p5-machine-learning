#!/usr/bin/env python3
"""Build paired Markdown/LaTeX reports and compile/verify the PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.reporting import build_reports, compile_and_verify_pdf


if __name__ == "__main__":
    result = {"sources": build_reports(), "pdf": compile_and_verify_pdf()}
    print(json.dumps(result, indent=2, default=str))
