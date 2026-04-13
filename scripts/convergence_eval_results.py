#!/usr/bin/env python3
"""Show eval results for all intermediate checkpoints to assess convergence.

For each after_task_XX_step_YY checkpoint (all steps, not just the last):
  - on_task:  performance on the task trained in that phase (last in tasks_trained)
  - off_task: performance on all previously trained tasks (forgetting signal)

Results are grouped by task and printed in step order so you can see
how performance evolves across training steps within each task phase.

Usage:
    python scripts/convergence_eval_results.py <eval_dir>

    e.g. python scripts/convergence_eval_results.py \
             checkpoints/sft_checkpoints/huihan_run_0/evals
"""

import argparse
import json
import pathlib
import re
import sys


def load_phase(phase_dir: pathlib.Path) -> dict:
    results_file = phase_dir / "results.json"
    if not results_file.exists():
        return None
    return json.loads(results_file.read_text())


def compile(eval_dir: pathlib.Path) -> list[dict]:
    """Returns a list of task groups, each with all checkpoints sorted by step."""
    phase_dirs = sorted(eval_dir.glob("after_task_*"))
    if not phase_dirs:
        print(f"No after_task_* directories found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    # Group by task key, collecting all step variants.
    task_groups: dict[str, list[tuple[int, pathlib.Path]]] = {}
    for phase_dir in phase_dirs:
        m = re.match(r'(after_task_\w+?)(?:_step_(\d+))?$', phase_dir.name)
        if not m:
            continue
        task_key = m.group(1)
        step = int(m.group(2)) if m.group(2) is not None else -1
        task_groups.setdefault(task_key, []).append((step, phase_dir))

    groups = []
    for task_key in sorted(task_groups):
        checkpoints = sorted(task_groups[task_key], key=lambda x: x[0])
        rows = []
        for step, phase_dir in checkpoints:
            data = load_phase(phase_dir)
            if data is None:
                print(f"  Skipping {phase_dir.name}: no results.json", file=sys.stderr)
                continue

            tasks_trained = data["tasks_trained"]
            current_task  = tasks_trained[-1]
            prior_tasks   = tasks_trained[:-1]
            task_results  = data["tasks"]

            on  = task_results.get(current_task)
            on_rate = on["success_rate"] if on else None

            prior_rates = [task_results[t]["success_rate"] for t in prior_tasks if t in task_results]
            off_rate = round(sum(prior_rates) / len(prior_rates), 4) if prior_rates else None

            rows.append({
                "step": step,
                "phase": phase_dir.name,
                "task_name": current_task,
                "on_task": on_rate,
                "off_task": off_rate,
            })

        if rows:
            groups.append({"task_key": task_key, "task_name": rows[0]["task_name"], "checkpoints": rows})

    return groups


def print_summary(eval_dir: pathlib.Path, groups: list[dict]):
    print(f"\n{eval_dir}")
    for group in groups:
        print(f"\n  Task: {group['task_name']}")
        print(f"  {'Step':<10} {'On-task':>8} {'Off-task':>9}  Phase")
        print("  " + "-" * 70)
        for r in group["checkpoints"]:
            step_str = str(r["step"]) if r["step"] >= 0 else "(bare)"
            on_str   = f"{r['on_task']:.0%}"  if r["on_task"]  is not None else "n/a"
            off_str  = f"{r['off_task']:.0%}" if r["off_task"] is not None else "n/a"
            print(f"  {step_str:<10} {on_str:>8} {off_str:>9}  {r['phase']}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=pathlib.Path, help="Path to the evals/ directory")
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    if not eval_dir.is_dir():
        print(f"Error: {eval_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    groups = compile(eval_dir)
    print_summary(eval_dir, groups)


if __name__ == "__main__":
    main()
