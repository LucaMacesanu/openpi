#!/usr/bin/env bash
# eval_libero_suites.sh — Evaluate all standard LIBERO suites except libero_90.
#
# This script runs examples/libero/main.py sequentially for:
#   - libero_spatial
#   - libero_object
#   - libero_goal
#   - libero_10
#
# The policy server must already be running.
#
# Usage:
#   bash shells/eval_libero_suites.sh \
#       --host 127.0.0.1 \
#       --port 8000 \
#       --num_trials 10 \
#       --output_dir data/libero/evals/my_run

set -u
set -o pipefail

HOST="127.0.0.1"
PORT=8000
NUM_TRIALS=50
NUM_STEPS_WAIT=10
RESIZE_SIZE=224
REPLAN_STEPS=5
SEED=7
OUTPUT_DIR=""

SUITES=(
    "libero_spatial"
    "libero_object"
    "libero_goal"
    "libero_10"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)           HOST="$2";           shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --num_trials)     NUM_TRIALS="$2";     shift 2 ;;
        --num_steps_wait) NUM_STEPS_WAIT="$2"; shift 2 ;;
        --resize_size)    RESIZE_SIZE="$2";    shift 2 ;;
        --replan_steps)   REPLAN_STEPS="$2";   shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_ACTIVATE="$OPENPI_ROOT/examples/libero/.venv/bin/activate"

if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="$OPENPI_ROOT/data/libero/evals/$TIMESTAMP"
fi

mkdir -p "$OUTPUT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" == "libero" ]]; then
    echo "Using active conda environment: libero"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate libero
        echo "Activated conda environment: libero"
    else
        echo "Found 'conda' but could not locate conda.sh under: $CONDA_BASE" >&2
        exit 1
    fi
elif [[ -f "$VENV_ACTIVATE" ]]; then
    # Fallback for setups that still use the README's local virtualenv.
    # shellcheck disable=SC1091
    source "$VENV_ACTIVATE"
    echo "Activated virtualenv: $VENV_ACTIVATE"
else
    echo "Could not activate the LIBERO environment." >&2
    echo "Expected either an active 'libero' conda env, a working 'conda activate libero'," >&2
    echo "or a local virtualenv at: $VENV_ACTIVATE" >&2
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"

SUMMARY_TSV="$OUTPUT_DIR/summary.tsv"
RUN_INFO="$OUTPUT_DIR/run_info.txt"

cat > "$RUN_INFO" <<EOF
output_dir=$OUTPUT_DIR
host=$HOST
port=$PORT
num_trials=$NUM_TRIALS
num_steps_wait=$NUM_STEPS_WAIT
resize_size=$RESIZE_SIZE
replan_steps=$REPLAN_STEPS
seed=$SEED
suites=${SUITES[*]}
started_at=$(date --iso-8601=seconds)
EOF

printf "suite\tstatus\tsuccess_rate\ttotal_episodes\tlog_file\tvideo_dir\n" > "$SUMMARY_TSV"

echo "============================================================"
echo "LIBERO suite eval"
echo "Output dir : $OUTPUT_DIR"
echo "Server     : $HOST:$PORT"
echo "Suites     : ${SUITES[*]}"
echo "Trials     : $NUM_TRIALS"
echo "============================================================"

FAILED_SUITES=()

for SUITE in "${SUITES[@]}"; do
    SUITE_DIR="$OUTPUT_DIR/$SUITE"
    VIDEO_DIR="$SUITE_DIR/videos"
    LOG_FILE="$SUITE_DIR/eval.log"

    mkdir -p "$VIDEO_DIR"

    echo ""
    echo "------------------------------------------------------------"
    echo "Running suite: $SUITE"
    echo "Log file    : $LOG_FILE"
    echo "Video dir   : $VIDEO_DIR"
    echo "------------------------------------------------------------"

    CMD=(
        python "$OPENPI_ROOT/examples/libero/main.py"
        --args.host "$HOST"
        --args.port "$PORT"
        --args.task-suite-name "$SUITE"
        --args.num-trials-per-task "$NUM_TRIALS"
        --args.num-steps-wait "$NUM_STEPS_WAIT"
        --args.resize-size "$RESIZE_SIZE"
        --args.replan-steps "$REPLAN_STEPS"
        --args.video-out-path "$VIDEO_DIR"
        --args.seed "$SEED"
    )

    {
        echo "[eval_libero_suites] started_at=$(date --iso-8601=seconds)"
        echo "[eval_libero_suites] command=${CMD[*]}"
        echo ""
        "${CMD[@]}"
    } 2>&1 | tee "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}

    SUCCESS_RATE="$(grep -E "Total success rate:" "$LOG_FILE" | tail -1 | sed 's/.*Total success rate: //')"
    TOTAL_EPISODES="$(grep -E "Total episodes:" "$LOG_FILE" | tail -1 | sed 's/.*Total episodes: //')"

    if [[ -z "$SUCCESS_RATE" ]]; then
        SUCCESS_RATE="N/A"
    fi
    if [[ -z "$TOTAL_EPISODES" ]]; then
        TOTAL_EPISODES="N/A"
    fi

    if [[ $STATUS -eq 0 ]]; then
        SUITE_STATUS="ok"
    else
        SUITE_STATUS="fail"
        FAILED_SUITES+=("$SUITE")
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$SUITE" \
        "$SUITE_STATUS" \
        "$SUCCESS_RATE" \
        "$TOTAL_EPISODES" \
        "$LOG_FILE" \
        "$VIDEO_DIR" \
        >> "$SUMMARY_TSV"
done

echo "finished_at=$(date --iso-8601=seconds)" >> "$RUN_INFO"

echo ""
echo "============================================================"
echo "All requested suites finished."
echo "Summary    : $SUMMARY_TSV"
echo "Run info   : $RUN_INFO"
echo "Output dir : $OUTPUT_DIR"
echo "============================================================"

if [[ ${#FAILED_SUITES[@]} -gt 0 ]]; then
    echo "Failed suites: ${FAILED_SUITES[*]}" >&2
    exit 1
fi
