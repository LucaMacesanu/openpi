#!/usr/bin/env bash
# eval_run_parallel.sh — Evaluate all task checkpoints in an SFT run folder with concurrency control.
#
# Each (task, step) checkpoint gets its own policy server on a unique port:
#   PORT = BASE_PORT + checkpoint_index
#
# Concurrency is controlled by --max_parallel (default: number of GPUs in --cuda_devices).
# At most MAX_PARALLEL servers run simultaneously; remaining checkpoints queue and
# start as slots free up. Each active server is pinned to a GPU via round-robin.
#
# Usage:
#   bash shells/eval_run_parallel.sh \
#       --run_dir        checkpoints/sft_checkpoints/sft_run_0 \
#       --num_trials     10 \
#       --cuda_devices   0,1,2,3 \
#       [--max_parallel  3]    # default: 3
#       [--base_port     8000] \
#       [--stagger_delay 30]   # seconds between consecutive server starts within a batch

set -euo pipefail

RUN_DIR=""
NUM_TRIALS=10
HOST="0.0.0.0"
BASE_PORT=$(( 8000 + (${SLURM_JOB_ID:-0} % 50000) % 10000 ))
CUDA_DEVICES="0"
SERVER_TIMEOUT=900
STAGGER_DELAY=30  # stagger within a concurrent batch to avoid BLAS init races
MAX_PARALLEL=""   # empty = auto (set to N_GPUS after parsing)
FORCE=0           # if 1, skip the already-done check and re-evaluate everything

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)        RUN_DIR="$2";        shift 2 ;;
        --num_trials)     NUM_TRIALS="$2";     shift 2 ;;
        --host)           HOST="$2";           shift 2 ;;
        --base_port)      BASE_PORT="$2";      shift 2 ;;
        --cuda_devices)   CUDA_DEVICES="$2";   shift 2 ;;
        --server_timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
        --stagger_delay)  STAGGER_DELAY="$2";  shift 2 ;;
        --max_parallel)   MAX_PARALLEL="$2";   shift 2 ;;
        --force)          FORCE=1;             shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: $0 --run_dir <path> [--num_trials N] [--cuda_devices X,Y,Z] [--max_parallel N] [--base_port P]"
    exit 1
fi

# Parse GPU list for per-server pinning.
IFS=',' read -ra GPU_LIST <<< "$CUDA_DEVICES"
N_GPUS=${#GPU_LIST[@]}

# Default MAX_PARALLEL to the number of GPUs the process can actually see.
# Uses torch.cuda.device_count() so it respects CUDA_VISIBLE_DEVICES (set by SLURM),
# rather than relying on what was passed to --cuda_devices.
if [[ -z "$MAX_PARALLEL" ]]; then
    MAX_PARALLEL=3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect task checkpoint dirs in sorted order (task_00, task_01, ...).
mapfile -t TASK_DIRS < <(find "$RUN_DIR" -maxdepth 1 -type d -name "task_*" | sort)

if [[ ${#TASK_DIRS[@]} -eq 0 ]]; then
    echo "No task_* directories found in $RUN_DIR"
    exit 1
fi

echo "Found ${#TASK_DIRS[@]} task checkpoint(s) in $RUN_DIR"
echo "[eval_run_parallel] Base port:    $BASE_PORT (SLURM_JOB_ID=${SLURM_JOB_ID:-unset})"
echo "[eval_run_parallel] GPUs:         ${GPU_LIST[*]}"
echo "[eval_run_parallel] Max parallel: $MAX_PARALLEL"

# ---------------------------------------------------------------------------
# Build list of all (task_dir, step_dir, port) tuples, skipping done ones.
# ---------------------------------------------------------------------------
declare -a ALL_TASK_DIRS=()
declare -a ALL_STEP_DIRS=()
declare -a ALL_PORTS=()
declare -a ALL_CONFIGS=()

IDX=0
for TASK_DIR in "${TASK_DIRS[@]}"; do
    METADATA_FILE="$TASK_DIR/metadata.json"
    if [[ -f "$METADATA_FILE" ]]; then
        POLICY_CONFIG=$(python3 -c "import json; d=json.load(open('$METADATA_FILE')); print(d['config']['name'])" 2>/dev/null || echo "pi05_libero_sft")
        NUM_TRAINED=$(python3 -c "import json; d=json.load(open('$METADATA_FILE')); print(d['num_tasks_trained'])")
        BASE_EVAL_PHASE=$(python3 -c "print(f'after_task_{int(\"$NUM_TRAINED\") - 1:02d}')")
    else
        POLICY_CONFIG="pi05_libero_sft"
        BASE_EVAL_PHASE=""
    fi

    mapfile -t STEP_DIRS < <(find "$TASK_DIR" -maxdepth 1 -mindepth 1 -type d \
        | grep -E '/[0-9]+$' | sort -V)

    if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
        echo "WARNING: No numeric step dirs found in $TASK_DIR — skipping."
        continue
    fi

    for STEP_DIR in "${STEP_DIRS[@]}"; do
        STEP=$(basename "$STEP_DIR")

        # Skip already-complete evals (unless --force).
        if [[ $FORCE -eq 0 && -n "$BASE_EVAL_PHASE" ]]; then
            EVAL_PHASE="${BASE_EVAL_PHASE}_step_${STEP}"
            RESULTS_FILE="$RUN_DIR/evals/$EVAL_PHASE/results.json"
            if [[ -f "$RESULTS_FILE" ]]; then
                NUM_RESULTS=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(len(d.get('tasks', {})))" 2>/dev/null || echo "0")
                NUM_EXPECTED=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(len(d.get('tasks_trained', [])))" 2>/dev/null || echo "0")
                if [[ "$NUM_RESULTS" -gt 0 && "$NUM_RESULTS" -ge "$NUM_EXPECTED" ]]; then
                    echo "[eval_run_parallel] Skipping $(basename "$TASK_DIR") step=$STEP ($NUM_RESULTS/$NUM_EXPECTED tasks already done)."
                    continue
                else
                    echo "[eval_run_parallel] Re-evaluating $(basename "$TASK_DIR") step=$STEP (incomplete: $NUM_RESULTS/$NUM_EXPECTED tasks)."
                fi
            fi
        fi

        PORT=$(( BASE_PORT + IDX ))
        ALL_TASK_DIRS+=("$TASK_DIR")
        ALL_STEP_DIRS+=("$STEP_DIR")
        ALL_PORTS+=("$PORT")
        ALL_CONFIGS+=("$POLICY_CONFIG")
        IDX=$(( IDX + 1 ))
    done
done

TOTAL=${#ALL_STEP_DIRS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "[eval_run_parallel] All checkpoints already evaluated."
    exit 0
fi

echo ""
echo "[eval_run_parallel] $TOTAL checkpoint(s) to evaluate, $MAX_PARALLEL slot(s) available."
echo ""

# ---------------------------------------------------------------------------
# Semaphore: track active subshell PIDs. Before each launch, poll until the
# number of running subshells drops below MAX_PARALLEL, then take a slot.
#
# GPU assignment: launch number LAUNCH_NUM gets GPU_LIST[LAUNCH_NUM % N_GPUS],
# so slots round-robin across available devices.
# ---------------------------------------------------------------------------

# Removes finished PIDs from ACTIVE_PIDS (modifies global array).
reap_finished() {
    local still_running=()
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        if kill -0 "$pid" 2>/dev/null; then
            still_running+=("$pid")
        fi
    done
    ACTIVE_PIDS=("${still_running[@]+"${still_running[@]}"}")
}

# Block until a slot is free, then return.
wait_for_slot() {
    while true; do
        reap_finished
        if [[ ${#ACTIVE_PIDS[@]} -lt $MAX_PARALLEL ]]; then
            return
        fi
        sleep 5
    done
}

declare -a ACTIVE_PIDS=()   # currently running subshell PIDs
declare -a ALL_PIDS=()       # every subshell PID, for final reporting
LAUNCH_NUM=0

for i in $(seq 0 $(( TOTAL - 1 ))); do
    TASK_DIR="${ALL_TASK_DIRS[$i]}"
    STEP_DIR="${ALL_STEP_DIRS[$i]}"
    PORT="${ALL_PORTS[$i]}"
    CONFIG="${ALL_CONFIGS[$i]}"
    STEP=$(basename "$STEP_DIR")
    LABEL="$(basename "$TASK_DIR") step=$STEP port=$PORT"

    # Block until a slot is free.
    wait_for_slot

    # Round-robin GPU assignment based on launch order.
    GPU="${GPU_LIST[$((LAUNCH_NUM % N_GPUS))]}"
    SLOT_STAGGER=$(( (LAUNCH_NUM % MAX_PARALLEL) * STAGGER_DELAY ))
    LAUNCH_NUM=$(( LAUNCH_NUM + 1 ))

    echo "[eval_run_parallel] Launching slot $LAUNCH_NUM/$TOTAL: $LABEL (GPU=$GPU)"

    (
        set -euo pipefail

        # Small stagger within a batch to avoid simultaneous BLAS initialization.
        if [[ $SLOT_STAGGER -gt 0 ]]; then
            sleep "$SLOT_STAGGER"
        fi

        echo "[parallel $PORT] Starting server on GPU $GPU: $LABEL"

        export XLA_PYTHON_CLIENT_PREALLOCATE=false

        SERVER_PID=$(bash "$SCRIPT_DIR/serve_policy.sh" \
            --checkpoint_dir "$STEP_DIR" \
            --cuda_devices   "$GPU" \
            --host           "$HOST" \
            --port           "$PORT" \
            --timeout        "$SERVER_TIMEOUT" \
            --config         "$CONFIG")

        echo "[parallel $PORT] Server up (PID=$SERVER_PID) — running eval: $LABEL"

        server_cleanup() {
            kill "$SERVER_PID" 2>/dev/null || true
        }
        trap server_cleanup EXIT

        bash "$SCRIPT_DIR/eval_checkpoint.sh" \
            --checkpoint_dir "$TASK_DIR" \
            --exp_dir        "$RUN_DIR" \
            --num_trials     "$NUM_TRIALS" \
            --host           "$HOST" \
            --port           "$PORT" \
            --step           "$STEP"

        echo "[parallel $PORT] Eval complete: $LABEL"
    ) &

    PID=$!
    ACTIVE_PIDS+=("$PID")
    ALL_PIDS+=("$PID")
done

# ---------------------------------------------------------------------------
# Wait for all remaining subshells; report failures.
# ---------------------------------------------------------------------------
EXIT_CODE=0
for i in "${!ALL_PIDS[@]}"; do
    PID="${ALL_PIDS[$i]}"
    TASK_DIR="${ALL_TASK_DIRS[$i]}"
    STEP=$(basename "${ALL_STEP_DIRS[$i]}")
    PORT="${ALL_PORTS[$i]}"
    if wait "$PID"; then
        echo "[eval_run_parallel] Done:   $(basename "$TASK_DIR") step=$STEP (port=$PORT)"
    else
        echo "[eval_run_parallel] FAILED: $(basename "$TASK_DIR") step=$STEP (port=$PORT)"
        EXIT_CODE=1
    fi
done

echo ""
echo "All parallel evals finished. Results in: $RUN_DIR/evals/"

echo "Compiling eval summary..."
uv run scripts/compile_eval_results.py "$RUN_DIR/evals"

exit $EXIT_CODE
