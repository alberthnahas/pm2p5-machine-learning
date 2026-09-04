#!/usr/bin/env python3
"""Run one prepared issue-time forecast using the deployment bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.deployment import run_operational_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--issue-time",
        help="UTC ISO issue time with timezone (default: latest common prepared time)",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    args = parser.parse_args()
    _, metadata = run_operational_forecast(args.issue_time, args.output)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
