"""Print a Sequential Task Evaluation Matrix as ASCII art to a text file.

Rows    = training checkpoint (after task i)
Columns = evaluation task j
Cell    = success rate of checkpoint i on task j
Cells where j > i are masked (task not yet seen).

Usage
-----
# From real results:
python scripts/print_eval_matrix.py --run_dir checkpoints/sft_checkpoints/sft_run_0

# Dummy data only (no run_dir needed):
python scripts/print_eval_matrix.py --dummy --num_tasks 6

# Save to file instead of stdout:
python scripts/print_eval_matrix.py --run_dir ... --output figures/matrix.txt
"""

import argparse
import json
import pathlib
import random
import re

import numpy as np


# ---------------------------------------------------------------------------
# Data loading (identical to plot_eval_matrix.py)
# ---------------------------------------------------------------------------

def load_run_matrix(run_dir: pathlib.Path):
    evals_dir = run_dir / "evals"
    phase_dirs = sorted(evals_dir.glob("after_task_*"))
    if not phase_dirs:
        raise FileNotFoundError(f"No after_task_* dirs found in {evals_dir}")

    phases = {}
    for phase_dir in phase_dirs:
        results_file = phase_dir / "results.json"
        if not results_file.exists():
            continue
        data = json.loads(results_file.read_text())
        m = re.search(r"after_task_(\d+)", data["eval_phase"])
        if m is None:
            continue
        phases[int(m.group(1))] = data

    if not phases:
        raise FileNotFoundError(f"No valid results.json files found under {evals_dir}")

    num_checkpoints = max(phases.keys()) + 1
    final_phase = phases[max(phases.keys())]
    task_labels = final_phase["tasks_trained"]
    num_tasks = len(task_labels)
    task_index = {name: i for i, name in enumerate(task_labels)}

    matrix = np.full((num_checkpoints, num_tasks), np.nan)
    for phase_idx, data in phases.items():
        for task_name, task_result in data["tasks"].items():
            j = task_index.get(task_name)
            if j is not None:
                matrix[phase_idx, j] = task_result["success_rate"]

    return matrix, task_labels


def make_dummy_matrix(num_tasks: int, seed: int = 42):
    rng = random.Random(seed)
    matrix = np.full((num_tasks, num_tasks), np.nan)
    base = [rng.uniform(0.55, 0.95) for _ in range(num_tasks)]
    forget = [rng.uniform(0.02, 0.12) for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(i + 1):
            steps_since = i - j
            sr = base[j] - forget[j] * steps_since + rng.gauss(0, 0.03)
            matrix[i, j] = float(np.clip(sr, 0.0, 1.0))
    labels = [f"Task {j:02d}" for j in range(num_tasks)]
    return matrix, labels


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

CELL_W = 6   # characters wide per cell (including padding), must be even

def render_matrix(matrix: np.ndarray, task_labels, title: str) -> str:
    num_checkpoints, num_tasks = matrix.shape

    # Shorten labels to fit cell width
    col_w = max(CELL_W, 5)
    short_labels = []
    for lbl in task_labels:
        if len(lbl) > col_w:
            short_labels.append(lbl[: col_w - 1] + "~")
        else:
            short_labels.append(lbl)

    row_label_w = len(f"After task {num_checkpoints - 1:02d}")

    def hline(left, mid, right, fill="-"):
        inner = (fill * col_w + mid) * num_tasks
        inner = inner[: -len(mid)]   # drop trailing mid
        return left + fill * row_label_w + mid + inner + right

    lines = []

    # Title
    lines.append(title)
    lines.append("")

    # Column header
    header_pad = " " * row_label_w
    col_headers = "|".join(lbl.center(col_w) for lbl in short_labels)
    lines.append(f"{header_pad} | {col_headers} |")
    lines.append(hline("+", "+", "+", "-"))

    # Rows
    for i in range(num_checkpoints):
        row_label = f"After task {i:02d}"
        cells = []
        for j in range(num_tasks):
            if np.isnan(matrix[i, j]):
                cells.append(" " * col_w)
            else:
                pct = f"{matrix[i, j]:.0%}"
                cells.append(pct.center(col_w))
        line = f"{row_label} | {'|'.join(cells)} |"
        lines.append(line)
        lines.append(hline("+", "+", "+", "-"))

    lines.append("")
    lines.append("(blank = task not yet seen at that checkpoint)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=pathlib.Path, default=None,
                        help="Path to the SFT run directory (must contain evals/)")
    parser.add_argument("--dummy", action="store_true",
                        help="Use generated dummy data instead of real results")
    parser.add_argument("--num_tasks", type=int, default=6,
                        help="Number of tasks for dummy data (default: 6)")
    parser.add_argument("--title", type=str, default="Sequential Task Evaluation Matrix",
                        help="Table title")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Save to this .txt path instead of printing to stdout")
    args = parser.parse_args()

    if args.dummy:
        matrix, labels = make_dummy_matrix(args.num_tasks)
        title = f"{args.title} (dummy data, {args.num_tasks} tasks)"
        default_output = None
    elif args.run_dir is not None:
        matrix, labels = load_run_matrix(args.run_dir)
        title = f"{args.title} -- {args.run_dir.name}"
        default_output = args.run_dir / "eval_matrix.txt"
    else:
        parser.error("Provide --run_dir or --dummy")

    output = args.output if args.output is not None else default_output
    text = render_matrix(matrix, labels, title)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"Saved to {output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
