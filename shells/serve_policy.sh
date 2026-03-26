#!/usr/bin/env bash
# serve_policy.sh — Start the openpi policy server for a checkpoint and wait until ready.
#
# Starts serve_policy.py in the background, polls until the port is accepting
# connections, then prints the server PID to stdout so the caller can kill it.
#
# Usage:
#   SERVER_PID=$(bash shells/serve_policy.sh \
#       --checkpoint_dir checkpoints/sft_checkpoints/sft_run_0/task_01_.../4999 \
#       --cuda_devices 4,6,7 \
#       --port 8000)
#   # ... do work ...
#   kill "$SERVER_PID"

set -euo pipefail

CHECKPOINT_DIR=""
CUDA_DEVICES="0"
HOST="0.0.0.0"
PORT=8000
POLL_INTERVAL=5   # seconds between readiness checks
TIMEOUT=300       # seconds to wait before giving up
CONFIG="pi05_libero_sft"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --cuda_devices)   CUDA_DEVICES="$2";   shift 2 ;;
        --host)           HOST="$2";           shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --timeout)        TIMEOUT="$2";        shift 2 ;;
        --config)         CONFIG="$2";         shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "Usage: $0 --checkpoint_dir <step_dir> [--cuda_devices X,Y] [--host H] [--port P]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_FILE="$(mktemp /tmp/serve_policy_XXXXXX.log)"

# All status messages go to stderr so only the PID reaches stdout.
echo "[serve_policy] Checkpoint : $CHECKPOINT_DIR" >&2
echo "[serve_policy] CUDA devices: $CUDA_DEVICES"  >&2
echo "[serve_policy] Port       : $PORT"            >&2
echo "[serve_policy] Log        : $LOG_FILE"        >&2

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    uv run "$OPENPI_ROOT/scripts/serve_policy.py" \
        policy:checkpoint \
        --policy.config="$CONFIG" \
        --policy.dir="$CHECKPOINT_DIR" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "[serve_policy] Started server (PID=$SERVER_PID)" >&2

# Poll until the port is open or we time out.
ELAPSED=0
until nc -z "$HOST" "$PORT" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[serve_policy] ERROR: server process died. Last log:" >&2
        tail -20 "$LOG_FILE" >&2
        exit 1
    fi
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[serve_policy] ERROR: timed out waiting for server on port $PORT after ${TIMEOUT}s" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

echo "[serve_policy] Server ready on port $PORT (${ELAPSED}s)" >&2

# Print only the PID to stdout for the caller to capture.
echo "$SERVER_PID"
