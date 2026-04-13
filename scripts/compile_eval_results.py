#!/usr/bin/env python3
"""Compile per-phase eval results into a single summary.

For each after_task_XX phase:
  - on_task:  performance on the task trained in that phase (last in tasks_trained)
  - off_task: performance on all previously trained tasks (forgetting signal)

Usage:
    python scripts/compile_eval_results.py <eval_dir>

    e.g. python scripts/compile_eval_results.py \
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


def compile(eval_dir: pathlib.Path) -> dict:
    phase_dirs = sorted(eval_dir.glob("after_task_*"))
    if not phase_dirs:
        print(f"No after_task_* directories found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    # Group dirs by task key (e.g. "after_task_00"), keep only the last checkpoint.
    # For step-specific dirs, "last" = highest step number; bare dirs are used only
    # when no step-specific sibling exists.
    task_groups: dict[str, pathlib.Path] = {}
    for phase_dir in phase_dirs:
        m = re.match(r'(after_task_\w+?)(?:_step_(\d+))?$', phase_dir.name)
        if not m:
            continue
        task_key = m.group(1)
        step = int(m.group(2)) if m.group(2) is not None else -1

        existing = task_groups.get(task_key)
        if existing is None:
            task_groups[task_key] = (step, phase_dir)
        else:
            existing_step, _ = existing
            if step > existing_step:
                task_groups[task_key] = (step, phase_dir)

    selected_dirs = [phase_dir for _, phase_dir in sorted(task_groups.values(), key=lambda x: x[1].name)]

    rows = []
    for phase_dir in selected_dirs:
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
            "task_num": len(rows),  # contiguous numbering regardless of skips
            "task_name": current_task,
            "phase": phase_dir.name,
            "on_task": on_rate,
            "off_task": off_rate,
        })

    on_vals  = [r["on_task"]  for r in rows if r["on_task"]  is not None]
    off_vals = [r["off_task"] for r in rows if r["off_task"] is not None]
    return {
        "eval_dir": str(eval_dir),
        "avg_on_task":  round(sum(on_vals)  / len(on_vals),  4) if on_vals  else None,
        "avg_off_task": round(sum(off_vals) / len(off_vals), 4) if off_vals else None,
        "results": rows,
    }


def print_summary(summary: dict):
    print(f"\n{summary['eval_dir']}")
    print(f"{'#':<4} {'Task':<58} {'On-task':>8} {'Off-task':>9}")
    print("-" * 82)
    for r in summary["results"]:
        on_str  = f"{r['on_task']:.0%}"  if r["on_task"]  is not None else "n/a"
        off_str = f"{r['off_task']:.0%}" if r["off_task"] is not None else "n/a"
        print(f"{r['task_num']:<4} {r['task_name'][:57]:<58} {on_str:>8} {off_str:>9}")

    on_vals  = [r["on_task"]  for r in summary["results"] if r["on_task"]  is not None]
    off_vals = [r["off_task"] for r in summary["results"] if r["off_task"] is not None]
    avg_on  = sum(on_vals)  / len(on_vals)  if on_vals  else None
    avg_off = sum(off_vals) / len(off_vals) if off_vals else None
    on_str  = f"{avg_on:.0%}"  if avg_on  is not None else "n/a"
    off_str = f"{avg_off:.0%}" if avg_off is not None else "n/a"
    print("-" * 82)
    print(f"{'avg':<4} {'':<58} {on_str:>8} {off_str:>9}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=pathlib.Path, help="Path to the evals/ directory")
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    if not eval_dir.is_dir():
        print(f"Error: {eval_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    summary = compile(eval_dir)
    print_summary(summary)

    out = eval_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
