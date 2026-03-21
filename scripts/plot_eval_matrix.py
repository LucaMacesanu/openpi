"""Plot a Sequential Task Evaluation Matrix from SFT run eval results.

Rows    = training checkpoint (after task i)
Columns = evaluation task j
Cell    = success rate of checkpoint i on task j
Cells where j > i are masked (task not yet seen).

Usage
-----
# From real results:
python scripts/plot_eval_matrix.py --run_dir checkpoints/sft_checkpoints/sft_run_0

# Dummy data only (no run_dir needed):
python scripts/plot_eval_matrix.py --dummy --num_tasks 6

# Save to file instead of showing:
python scripts/plot_eval_matrix.py --run_dir ... --output figures/matrix.png
"""

import argparse
import json
import pathlib
import random
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_matrix(run_dir: pathlib.Path):
    """Return (matrix, task_labels) from a run's evals/ directory.

    matrix[i, j] = success rate of checkpoint after_task_i on task j.
    Entries where j > i are NaN (task not yet encountered).
    """
    evals_dir = run_dir / "evals"
    phase_dirs = sorted(evals_dir.glob("after_task_*"))
    if not phase_dirs:
        raise FileNotFoundError(f"No after_task_* dirs found in {evals_dir}")

    # Collect all results, keyed by phase index.
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

    # Build ordered task list from the final checkpoint's tasks_trained.
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
    """Generate plausible-looking dummy data for illustration."""
    rng = random.Random(seed)
    matrix = np.full((num_tasks, num_tasks), np.nan)

    # Simulated "base" competence per task and a forgetting rate.
    base = [rng.uniform(0.55, 0.95) for _ in range(num_tasks)]
    forget = [rng.uniform(0.02, 0.12) for _ in range(num_tasks)]

    for i in range(num_tasks):        # checkpoint index
        for j in range(i + 1):        # only tasks seen so far
            steps_since = i - j
            sr = base[j] - forget[j] * steps_since + rng.gauss(0, 0.03)
            matrix[i, j] = float(np.clip(sr, 0.0, 1.0))

    labels = [f"Task {j:02d}" for j in range(num_tasks)]
    return matrix, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def shorten(label: str, max_len: int = 32) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def _pastel_rdylgn():
    """Pastel red → yellow → green colormap (desaturated, easy on the eyes)."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        "#e8a0a0",   # pastel red
        "#f0d080",   # pastel yellow
        "#90c990",   # pastel green
    ]
    return LinearSegmentedColormap.from_list("pastel_rdylgn", colors)


def plot_matrix(
    matrix: np.ndarray,
    task_labels,
    title: str = "Sequential Task Evaluation Matrix",
    output=None,
) -> None:
    num_checkpoints, num_tasks = matrix.shape
    short_labels = [shorten(t) for t in task_labels]

    masked = np.ma.masked_invalid(matrix)

    cell_size = 1.4
    fig_w = max(7, num_tasks * cell_size + 3.5)
    fig_h = max(5, num_checkpoints * cell_size + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    cmap = _pastel_rdylgn()
    cmap.set_bad(color="#ebebeb")   # unseen cells → very light grey

    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto",
                   interpolation="nearest")

    # Cell annotations.
    for i in range(num_checkpoints):
        for j in range(num_tasks):
            if np.isnan(matrix[i, j]):
                continue
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color="#333333",
                    fontfamily="sans-serif")

    # Thin grid lines between cells.
    ax.set_xticks(np.arange(-0.5, num_tasks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_checkpoints, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # Diagonal boundary line (step-shape separating seen / unseen).
    n = min(num_checkpoints, num_tasks)
    for i in range(n):
        ax.plot([i + 0.5, i + 0.5], [i - 0.5, i + 0.5], color="#555555", lw=1.2)
        ax.plot([i - 0.5, i + 0.5], [i + 0.5, i + 0.5], color="#555555", lw=1.2)

    # Subtle diagonal highlight (no bevel — just a thin coloured border).
    for k in range(n):
        rect = matplotlib.patches.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            linewidth=1.8, edgecolor="#4a7fb5", facecolor="none",
        )
        ax.add_patch(rect)

    # Axes.
    ax.set_xticks(range(num_tasks))
    ax.set_xticklabels(short_labels, rotation=40, ha="right",
                       fontsize=8.5, color="#444444")
    ax.set_yticks(range(num_checkpoints))
    ax.set_yticklabels([f"After task {i:02d}" for i in range(num_checkpoints)],
                       fontsize=8.5, color="#444444")
    ax.set_xlabel("Evaluation Task", fontsize=10, labelpad=10, color="#333333")
    ax.set_ylabel("Training Checkpoint", fontsize=10, labelpad=10, color="#333333")
    ax.set_title(title, fontsize=13, pad=14, color="#222222", fontweight="bold")

    # Remove outer spines for a cleaner look.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colour bar — slim, no border.
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Success Rate", fontsize=9, color="#444444")
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    cbar.ax.tick_params(labelsize=8, colors="#555555")

    # Compact legend below the plot.
    legend_elements = [
        matplotlib.patches.Patch(facecolor="#ebebeb", edgecolor="#cccccc",
                                 label="Not yet trained"),
        matplotlib.patches.Patch(facecolor="none", edgecolor="#4a7fb5",
                                 linewidth=1.8, label="Immediate performance"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(0.0, -0.15), ncol=2,
              fontsize=8.5, framealpha=0, borderpad=0)

    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {output}")
    else:
        plt.show()


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
                        help="Plot title")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Save figure to this path instead of displaying it")
    args = parser.parse_args()

    if args.dummy:
        matrix, labels = make_dummy_matrix(args.num_tasks)
        title = f"{args.title} (dummy data, {args.num_tasks} tasks)"
        default_output = None
    elif args.run_dir is not None:
        matrix, labels = load_run_matrix(args.run_dir)
        title = f"{args.title} — {args.run_dir.name}"
        default_output = args.run_dir / "eval_matrix.png"
    else:
        parser.error("Provide --run_dir or --dummy")

    output = args.output if args.output is not None else default_output
    plot_matrix(matrix, labels, title=title, output=output)


if __name__ == "__main__":
    main()
