#!/usr/bin/env bash
# End-to-end orchestrator: pretrain → SFT → DPO → (REPL) for OpenMythos-3B.
#
# By default this script re-execs itself under nohup so the whole pipeline runs
# in the background. Logs land in ./logs/. The REPL is interactive and is
# skipped in background mode — attach manually afterwards.
#
# Usage:
#   bash runs/run_all.sh                       # default: nohup background
#   FOREGROUND=1 bash runs/run_all.sh          # run in the current shell
#   STAGES=sft,dpo bash runs/run_all.sh        # pick a subset
#   MYTHOS_TARGET_TOKENS=50000000 bash runs/run_all.sh   # smoke test
#
# After launch:
#   tail -f logs/run_all_*.log
#   tensorboard --logdir logs/tb --port 6006 --bind_all
#   python runs/eval_repl.py        # chat with the final checkpoint

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

STAGES="${STAGES:-pretrain,sft,dpo,repl}"
FOREGROUND="${FOREGROUND:-0}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
TB_DIR="${TB_DIR:-$LOG_DIR/tb}"
VENV="${VENV:-$REPO_DIR/.venv}"
TB_PORT="${TB_PORT:-6006}"
START_TB="${START_TB:-1}"

mkdir -p "$LOG_DIR" "$TB_DIR"

# --- Re-exec under nohup unless already detached or FOREGROUND=1 -----------
if [[ "${_NOHUP:-0}" != "1" && "$FOREGROUND" != "1" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="$LOG_DIR/run_all_${TS}.log"
    echo "[run_all] launching in background…"
    _NOHUP=1 nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
    PID=$!
    disown "$PID" 2>/dev/null || true
    echo "[run_all] PID=$PID"
    echo "[run_all] log=$LOG"
    echo "[run_all] tail:  tail -f $LOG"
    echo "[run_all] stop:  kill $PID"
    echo "[run_all] tb:    tensorboard --logdir $TB_DIR --port $TB_PORT --bind_all"
    echo "[run_all] repl:  python runs/eval_repl.py   (after stages finish)"
    exit 0
fi

# --- From here on we're the detached worker --------------------------------
echo "[run_all] === begin === $(date -Iseconds)"
echo "[run_all] pid=$$  repo=$REPO_DIR  stages=$STAGES"

# Activate the venv (created by runs/setup_env.sh)
if [[ -d "$VENV" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "[run_all] venv=$VENV  python=$(command -v python)"
else
    echo "[run_all] ERROR: no venv at $VENV — run 'bash runs/setup_env.sh' first" >&2
    exit 1
fi

NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)')}"
echo "[run_all] nproc=$NPROC"

# Optionally start a background tensorboard server
TB_PID=""
if [[ "$START_TB" == "1" ]]; then
    if command -v tensorboard >/dev/null 2>&1; then
        TB_LOG="$LOG_DIR/tensorboard.log"
        nohup tensorboard --logdir "$TB_DIR" --port "$TB_PORT" --bind_all \
            > "$TB_LOG" 2>&1 < /dev/null &
        TB_PID=$!
        disown "$TB_PID" 2>/dev/null || true
        echo "[run_all] tensorboard pid=$TB_PID  log=$TB_LOG  port=$TB_PORT"
    else
        echo "[run_all] tensorboard not on PATH — skipping server startup"
    fi
fi

run_stage() {
    local name="$1"
    local script="$2"
    if [[ ",$STAGES," != *",$name,"* ]]; then
        echo "[run_all] skip $name"
        return 0
    fi
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local stage_log="$LOG_DIR/${name}_${ts}.log"
    echo "[run_all] === $name ===  log=$stage_log"
    export MYTHOS_TB_DIR="$TB_DIR/$name"
    mkdir -p "$MYTHOS_TB_DIR"
    if [[ "$NPROC" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="$NPROC" "$script" \
            > "$stage_log" 2>&1
    else
        python "$script" > "$stage_log" 2>&1
    fi
    echo "[run_all] $name finished at $(date -Iseconds)"
}

run_stage pretrain runs/pretrain_3b.py
run_stage sft      runs/sft_3b.py
run_stage dpo      runs/dpo_3b.py

if [[ ",$STAGES," == *",repl,"* ]]; then
    if [[ "${_NOHUP:-0}" == "1" ]]; then
        echo "[run_all] repl is interactive — not launched in background."
        echo "[run_all] start it manually:  python runs/eval_repl.py"
    else
        echo "[run_all] === repl ==="
        python runs/eval_repl.py
    fi
fi

echo "[run_all] === end ===   $(date -Iseconds)"

if [[ -n "$TB_PID" ]]; then
    echo "[run_all] tensorboard still running at pid=$TB_PID (kill when done)"
fi
