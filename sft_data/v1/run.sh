#!/bin/bash
# SFT data v1 — full reproduction (SFT_DATA_SPEC.md).
# Requires analysis/phase0_5/v1.1/ (Phase 0.5 output) and the packages in
# sft_data/v1/src/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PY="${1:-python3}"

echo "== 1/2 unit tests"
"$PY" sft_data/v1/src/tests/test_build.py

echo "== 2/2 build (packed + prefix JSONL, manifests, leakage, REPORT.md)"
"$PY" sft_data/v1/src/build_sft.py

echo "Done. See sft_data/v1/REPORT.md"
