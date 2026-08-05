#!/usr/bin/env bash
# Server-side test runner for the NOD benchmark solution.
#
# Works on any Linux server (x86_64 or arm64 Jetson) that has python3.
# It creates an isolated venv, installs deps from requirements.txt, then runs
# the full local test suite (unit tests + synthetic self-eval + smoke test).
#
# Usage:
#   bash scripts/test_server.sh
#   PYTHON=python3.11 bash scripts/test_server.sh   # pin a python
set -e

PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

echo "== [1/4] Python: $($PY --version 2>&1) =="
echo "== [2/4] Creating venv + installing deps =="
if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== [3/4] Unit tests =="
python tests/test_geometric.py

echo "== [4/4] Synthetic self-eval (50 frames) =="
python -m nod.evaluate --demo 50

echo "== smoke: predict() on a flat 5 m scene =="
python -c "from solution import predict; import numpy as np; \
print('NOD =', round(predict({'depth': np.full((480,640),5.0,np.float32)}),3), 'm')"

echo "ALL GREEN ✅"
