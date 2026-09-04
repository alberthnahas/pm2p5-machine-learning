#!/usr/bin/env bash
set -euo pipefail
umask 027

experiment_root="/run/media/workstation-llm/HDD2/AQ/experiments/machine-learning"
python_bin="/run/media/workstation-llm/HDD2/.venv/bin/python"
mkdir -p "$experiment_root/shadow/logs"
export MPLCONFIGDIR=/tmp/matplotlib-aq
cd "$experiment_root"
exec "$python_bin" scripts/run_shadow_daily.py
