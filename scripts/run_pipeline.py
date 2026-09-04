#!/usr/bin/env python3
"""Run and resource-profile the deterministic post-acquisition experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
RESOURCE_PATH = PROVENANCE / "execution_resource_usage.json"
STAGES = [
    ("prepare", "prepare_data.py"),
    ("train", "train_evaluate.py"),
    ("diagnostics", "analyze_diagnostics.py"),
    ("deployment", "build_deployment.py"),
    ("figures", "make_figures.py"),
    ("notebook", "execute_notebook.py"),
    ("report", "build_report.py"),
]


def _write_resource(payload: dict) -> None:
    temporary = RESOURCE_PATH.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, RESOURCE_PATH)


def _rss_tree(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for child in processes:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_stage(name: str, script: str, resources: dict) -> None:
    stdout_path = PROVENANCE / f"{name}.stdout.log"
    stderr_path = PROVENANCE / f"{name}.stderr.log"
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/matplotlib-aq"
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    peak_rss = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        child = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        process = psutil.Process(child.pid)
        while child.poll() is None:
            peak_rss = max(peak_rss, _rss_tree(process))
            time.sleep(0.1)
        peak_rss = max(peak_rss, _rss_tree(process))
    elapsed = time.perf_counter() - started
    resources[name] = {
        "script": script,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "peak_rss_gib": peak_rss / (1024.0**3),
        "return_code": child.returncode,
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
    }
    _write_resource(resources)
    if child.returncode:
        raise RuntimeError(
            f"Stage {name} failed with return code {child.returncode}; "
            f"inspect {stderr_path}"
        )
    print(
        json.dumps(
            {
                "stage": name,
                "elapsed_seconds": round(elapsed, 2),
                "peak_rss_gib": round(peak_rss / (1024.0**3), 3),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-stage",
        choices=[name for name, _ in STAGES],
        default="prepare",
        help="Resume from this deterministic stage",
    )
    parser.add_argument(
        "--through-stage",
        choices=[name for name, _ in STAGES],
        default="report",
        help="Stop after this stage",
    )
    args = parser.parse_args()
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    resources = (
        json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))
        if RESOURCE_PATH.exists()
        else {}
    )
    names = [name for name, _ in STAGES]
    start = names.index(args.from_stage)
    end = names.index(args.through_stage)
    if end < start:
        raise ValueError("--through-stage precedes --from-stage")
    for name, script in STAGES[start : end + 1]:
        run_stage(name, script, resources)


if __name__ == "__main__":
    main()
