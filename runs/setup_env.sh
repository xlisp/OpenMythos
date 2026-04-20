#!/usr/bin/env bash
# Build a python venv at $REPO/.venv that inherits the system site-packages.
# The cluster's system python 3.12 already has torch 2.10.0+cu130 — we want to
# reuse it (no root, no pip-install-torch) and layer the project deps on top.
#
# Usage:
#     bash runs/setup_env.sh
#     source .venv/bin/activate
#
# Override:
#     PYTHON=/path/to/python3.12  VENV=/custom/.venv  bash runs/setup_env.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VENV="${VENV:-$REPO_DIR/.venv}"
PYTHON="${PYTHON:-python3}"

echo "[setup_env] repo=$REPO_DIR"
echo "[setup_env] python=$($PYTHON -V 2>&1)  path=$(command -v $PYTHON)"
echo "[setup_env] venv=$VENV"

if [[ ! -d "$VENV" ]]; then
    "$PYTHON" -m venv --system-site-packages "$VENV"
    echo "[setup_env] created venv (inherits system site-packages)"
else
    echo "[setup_env] venv already exists — reusing"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup_env] checking inherited torch / CUDA..."
python - <<'PY'
import torch
print(f"  torch:          {torch.__version__}")
print(f"  cuda.available: {torch.cuda.is_available()}")
print(f"  cuda.version:   {torch.version.cuda}")
print(f"  device_count:   {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    print(f"  device[0]:      {torch.cuda.get_device_name(0)}")
PY

# Upgrade pip inside the venv only. Do NOT touch torch — it is inherited from
# the system install with the correct CUDA 13 build.
python -m pip install --upgrade pip

python -m pip install -U \
    "datasets>=3.0.0" \
    "transformers>=4.40.0" \
    "loguru>=0.7.3" \
    "tensorboard>=2.15" \
    "huggingface_hub>=0.23"

echo "[setup_env] verifying imports..."
python - <<'PY'
import importlib
for m in ["torch", "datasets", "transformers", "loguru", "tensorboard"]:
    mod = importlib.import_module(m)
    print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
PY

echo
echo "[setup_env] done."
echo "  activate:    source $VENV/bin/activate"
echo "  run all:     bash runs/run_all.sh"
echo "  tb server:   tensorboard --logdir logs/tb --port 6006 --bind_all"
