#!/bin/bash
# ReviseQA Phase 0 — full reproduction from raw logs.
# Usage: bash analysis/phase0/run.sh [python]
# Requires the packages in analysis/phase0/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PY="${1:-python3}"

echo "== 1/3 coder unit tests"
"$PY" analysis/phase0/src/tests/test_coder.py

echo "== 2/3 build tidy table from raw logs + dataset"
"$PY" analysis/phase0/src/parse_logs.py

echo "== 3/3 descriptives, interaction tests, autopsy, figures, report"
"$PY" analysis/phase0/src/analysis.py

echo "Done. See analysis/phase0/report.md"
