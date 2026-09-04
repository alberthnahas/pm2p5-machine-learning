#!/usr/bin/env python3
"""Execute the audit notebook top-to-bottom in an isolated temporary kernel."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "pm25_station_forecast_audit.ipynb"
OUTPUT = ROOT / "notebooks" / "pm25_station_forecast_audit_executed.ipynb"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pm25-notebook-kernel-") as temporary:
        data_dir = Path(temporary)
        kernel_dir = data_dir / "kernels" / "pm25-local"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "PM2.5 HDD2 environment",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        previous_jupyter_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = str(data_dir)
        try:
            notebook = nbformat.read(SOURCE, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=300,
                kernel_name="pm25-local",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
            )
            executed = client.execute()
        finally:
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path
    nbformat.write(executed, OUTPUT)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errored = [
        cell
        for cell in code_cells
        if any(output.output_type == "error" for output in cell.get("outputs", []))
    ]
    manifest = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE.name,
        "output": OUTPUT.name,
        "code_cells": len(code_cells),
        "executed_code_cells": sum(cell.execution_count is not None for cell in code_cells),
        "errored_cells": len(errored),
        "output_bytes": OUTPUT.stat().st_size,
    }
    (ROOT / "provenance").mkdir(parents=True, exist_ok=True)
    (ROOT / "provenance" / "notebook_execution.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
