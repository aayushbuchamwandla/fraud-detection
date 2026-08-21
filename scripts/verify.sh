#!/bin/bash
# Full project verification. Run from the project root.
cd "$(dirname "$0")/.."
source scripts/wsl_env.sh 2>/dev/null || true
python scripts/verify.py
