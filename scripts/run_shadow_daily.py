#!/usr/bin/env python3
"""Run one idempotent PM2.5 shadow forecast and delayed verification cycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pm25ml.shadow import run_daily_shadow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--issue-date",
        help="Optional YYYY-MM-DD date for the validated 00 UTC cycle",
    )
    args = parser.parse_args()
    print(json.dumps(run_daily_shadow(args.issue_date), indent=2, default=str))


if __name__ == "__main__":
    main()
