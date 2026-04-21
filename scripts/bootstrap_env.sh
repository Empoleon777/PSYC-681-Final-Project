#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Python environment is ready."
echo "If you are using Postgres, run:"
echo "  python scripts/apply_schema.py"
echo "  python scripts/db_roundtrip_check.py"
