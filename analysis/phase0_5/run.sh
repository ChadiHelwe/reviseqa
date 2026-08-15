#!/bin/bash
# Phase 0.5 (SPEC_ADDENDUM_A §1) — full reproduction.
# Requires: analysis/phase0/tidy.parquet (run analysis/phase0/run.sh first),
# prover9 + mace4 binaries (brew install prover9), and the packages in
# analysis/phase0/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PY="${1:-python3}"

echo "== 0/4 FOL->LADR converter validation"
"$PY" analysis/phase0_5/src/ladr.py

echo "== 1/4 §1.1 comparator consistency audit"
"$PY" analysis/phase0_5/src/comparator_audit.py

echo "== 2/4 §1.2 transition split"
"$PY" analysis/phase0_5/src/transition_split.py

echo "== 3/4 §1.3 timeout-Uncertain audit"
"$PY" analysis/phase0_5/src/timeout_u_audit.py

echo "== 4/4 §1.4 v1.1 repair + freeze"
"$PY" analysis/phase0_5/src/repair_v1_1.py

echo "Done. See analysis/phase0_5/report.md and analysis/phase0_5/v1.1/"
