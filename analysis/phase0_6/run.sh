#!/bin/bash
# Phase 0.6 (PHASE0_6_RECUT.md) — full reproduction.
# Requires: analysis/phase0/tidy.parquet, analysis/phase0_5/v1.1/manifest.csv,
# analysis/phase0_5/tables/comparator_audit.csv, prover9+mace4 (Job C only),
# and the packages in analysis/phase0/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PY="${1:-python3}"

echo "== 1/2 Job C: leftover comparator pre-edit re-derivation (prover)"
"$PY" analysis/phase0_6/src/job_c.py

echo "== 2/2 Jobs A, B, D + report"
"$PY" analysis/phase0_6/src/recut.py

echo "Done. See analysis/phase0_6/report.md"
