#!/usr/bin/env bash
# End-to-end: pretrain → SFT → DPO → REPL for OpenMythos-3B on A800 x 8.
#
# Stages (select with $STAGES, default "pretrain,sft,dpo,repl"):
#   pretrain  — FineWeb-Edu pretraining (default 30B tokens; override via MYTHOS_TARGET_TOKENS)
#   sft       — Alpaca instruction tuning
#   dpo       — DPO on Intel/orca_dpo_pairs
#   repl      — launch interactive chat on the latest DPO checkpoint
#
# Example (smoke test, fewer tokens):
#   MYTHOS_TARGET_TOKENS=50000000 bash runs/run_all.sh
#
# Example (only SFT + DPO, pretrain already done):
#   STAGES=sft,dpo,repl bash runs/run_all.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)')}"
STAGES="${STAGES:-pretrain,sft,dpo,repl}"

echo "[run_all] repo=$REPO_DIR  nproc=$NPROC  stages=$STAGES"

run_stage() {
    local name="$1"
    local script="$2"
    if [[ ",$STAGES," != *",$name,"* ]]; then
        echo "[run_all] skip $name"
        return 0
    fi
    echo "[run_all] === $name ==="
    if [[ "$NPROC" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="$NPROC" "$script"
    else
        python "$script"
    fi
}

run_stage pretrain runs/pretrain_3b.py
run_stage sft      runs/sft_3b.py
run_stage dpo      runs/dpo_3b.py

if [[ ",$STAGES," == *",repl,"* ]]; then
    echo "[run_all] === repl ==="
    python runs/eval_repl.py
fi

echo "[run_all] complete."
