#!/usr/bin/env bash
# eval_checkpoint.sh — Evaluate all tasks a checkpoint was trained on.
#
# Reads tasks_trained.json from the given checkpoint directory, then runs
# libero_eval_task.py for each task, saving structured JSON results and videos
# under the experiment's evals/ folder.
#
# Usage:
#   bash shells/eval_checkpoint.sh \
#       --checkpoint_dir /path/to/experiments/sft_run_0/checkpoints/task_01_.../ \
#       --exp_dir /path/to/experiments/sft_run_0 \
#       --num_trials 10
#
# The policy server must already be running:
#   uv run scripts/serve_policy.py --env LIBERO --checkpoint <path>

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CHECKPOINT_DIR=""
EXP_DIR=""
NUM_TRIALS=10
HOST="0.0.0.0"
PORT=8000

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --exp_dir)        EXP_DIR="$2";        shift 2 ;;
        --num_trials)     NUM_TRIALS="$2";     shift 2 ;;
        --host)           HOST="$2";           shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" || -z "$EXP_DIR" ]]; then
    echo "Usage: $0 --checkpoint_dir <path> --exp_dir <path> [--num_trials N] [--host H] [--port P]"
    exit 1
fi

if [[ -f "$CHECKPOINT_DIR/metadata.json" ]]; then
    TASKS_JSON="$CHECKPOINT_DIR/metadata.json"
elif [[ -f "$CHECKPOINT_DIR/tasks_trained.json" ]]; then
    echo "Note: metadata.json not found, falling back to tasks_trained.json"
    TASKS_JSON="$CHECKPOINT_DIR/tasks_trained.json"
else
    echo "Error: neither metadata.json nor tasks_trained.json found in $CHECKPOINT_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse tasks_trained.json
# ---------------------------------------------------------------------------
NUM_TRAINED=$(python3 -c "import json; d=json.load(open('$TASKS_JSON')); print(d['num_tasks_trained'])")
EVAL_PHASE=$(python3 -c "print(f'after_task_{int(\"$NUM_TRAINED\") - 1:02d}')")
EVAL_DIR="$EXP_DIR/evals/$EVAL_PHASE"

echo "============================================================"
echo "Checkpoint : $CHECKPOINT_DIR"
echo "Tasks trained: $NUM_TRAINED"
echo "Eval phase : $EVAL_PHASE"
echo "Output dir : $EVAL_DIR"
echo "Trials/task: $NUM_TRIALS"
echo "============================================================"

mkdir -p "$EVAL_DIR"

# ---------------------------------------------------------------------------
# Activate libero venv
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$OPENPI_ROOT/examples/libero/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"

# ---------------------------------------------------------------------------
# Evaluate each trained task
# ---------------------------------------------------------------------------
NUM_TASKS=$(python3 -c "import json; d=json.load(open('$TASKS_JSON')); print(len(d['tasks_trained']))")

FAILED_TASKS=()
for i in $(seq 0 $((NUM_TASKS - 1))); do
    TASK=$(python3 -c "import json; d=json.load(open('$TASKS_JSON')); print(d['tasks_trained'][$i])")
    TASK_SLUG=$(echo "$TASK" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | cut -c1-60)

    echo ""
    echo "--- Task $((i+1))/$NUM_TASKS: $TASK ---"

    python "$OPENPI_ROOT/scripts/libero_eval_task.py" \
        --args.task "$TASK" \
        --args.num-trials "$NUM_TRIALS" \
        --args.host "$HOST" \
        --args.port "$PORT" \
        --args.video-out-path "$EVAL_DIR/videos/$TASK_SLUG" \
        --args.results-file "$EVAL_DIR/${TASK_SLUG}.json" \
        || { echo "WARNING: eval failed for task: $TASK"; FAILED_TASKS+=("$TASK"); }
done

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: ${#FAILED_TASKS[@]} task(s) failed and will be missing from results.json:"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
fi

# ---------------------------------------------------------------------------
# Merge per-task JSONs into a single results.json
# ---------------------------------------------------------------------------
python3 - <<EOF
import json, pathlib, datetime

eval_dir = pathlib.Path("$EVAL_DIR")
tasks_meta = json.loads(pathlib.Path("$TASKS_JSON").read_text())

tasks_results = {}
for f in sorted(eval_dir.glob("*.json")):
    if f.name == "results.json":
        continue
    d = json.loads(f.read_text())
    tasks_results[d["task"]] = d

results = {
    "eval_phase": "$EVAL_PHASE",
    "checkpoint_dir": "$CHECKPOINT_DIR",
    "tasks_trained": tasks_meta["tasks_trained"],
    "num_trials": $NUM_TRIALS,
    "timestamp": datetime.datetime.now().isoformat(),
    "tasks": tasks_results,
}
(eval_dir / "results.json").write_text(json.dumps(results, indent=2))
print("results.json written.")
EOF

echo ""
echo "============================================================"
echo "Eval complete. Results in: $EVAL_DIR/results.json"
echo "============================================================"
