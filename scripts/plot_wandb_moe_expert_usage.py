"""Plot concatenated MoE expert usage from a sequential SFT W&B run.

Example:
    .venv/bin/python scripts/plot_wandb_moe_expert_usage.py \
        --run_path tz2668-new-york-university/openpi/h13kr9pg \
        --num_tasks 10 \
        --num_experts 8 \
        --save_csv

    By default outputs go under:
        figures/<wandb run display name>/
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
import pathlib
import re
import sys

import numpy as np
import wandb


def _global_metric_name(task_idx: int, expert_idx: int) -> str:
    return f"task_{task_idx:02d}/moe/global_expert_{expert_idx}_usage"


def _layer_scalar_metric_name(task_idx: int, layer: int, expert_idx: int) -> str:
    return f"task_{task_idx:02d}/moe/layer_{layer}/expert_{expert_idx}_usage"


def _layer_vector_metric_name(task_idx: int, layer: int) -> str:
    return f"task_{task_idx:02d}/moe_layers/layer_{layer}/expert_usage"


def _target_slug(layer: int | None) -> str:
    if layer is None:
        return "global"
    return f"layer_{layer}"


def _target_title(layer: int | None) -> str:
    if layer is None:
        return "Global"
    return f"Layer {layer}"


def _safe_path_component(value: str | None) -> str:
    if not value:
        return "run"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "run"


def _default_run_output_dir(run, root_output_dir: pathlib.Path) -> pathlib.Path:
    return root_output_dir / _safe_path_component(getattr(run, "name", None) or getattr(run, "id", None))


def _default_output_name(layer: int | None, output_prefix: str | None) -> str:
    slug = _target_slug(layer)
    if output_prefix:
        return f"{output_prefix}_{slug}_expert_usage.png"
    return f"{slug}_expert_usage.png"


def _default_csv_name(layer: int | None, output_prefix: str | None) -> str:
    return pathlib.Path(_default_output_name(layer, output_prefix)).with_suffix(".csv").name


def _positive_step_delta(steps: np.ndarray) -> float:
    diffs = np.diff(np.unique(steps))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return 1.0
    return float(np.median(positive_diffs))


def _resolve_run(args: argparse.Namespace):
    api = wandb.Api()

    if args.run_path:
        return api.run(args.run_path)

    entity = args.entity or os.environ.get("WANDB_ENTITY") or getattr(api, "default_entity", None)
    if entity is None:
        raise ValueError("Provide --run_path, --entity, or set WANDB_ENTITY.")

    runs = list(api.runs(f"{entity}/{args.project}", filters={"display_name": args.run_name}))
    exact_matches = [run for run in runs if run.name == args.run_name]
    matches = exact_matches or runs
    if not matches:
        raise ValueError(f"No W&B runs found for display name {args.run_name!r} in {entity}/{args.project}.")

    matches.sort(key=lambda run: getattr(run, "created_at", "") or "", reverse=True)
    return matches[0]


def _coerce_usage_vector(value: object, num_experts: int) -> list[float] | None:
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, dict):
        return None

    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if array.size < num_experts:
        return None
    return array[:num_experts].tolist()


def _fetch_task_scalar_rows(
    run,
    *,
    task_idx: int,
    num_experts: int,
    page_size: int,
    layer: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if layer is None:
        metrics = [_global_metric_name(task_idx, expert_idx) for expert_idx in range(num_experts)]
    else:
        metrics = [_layer_scalar_metric_name(task_idx, layer, expert_idx) for expert_idx in range(num_experts)]
    rows = []

    for row in run.scan_history(keys=["_step", *metrics], page_size=page_size):
        if "_step" not in row:
            continue
        values = [row.get(metric) for metric in metrics]
        if any(value is None for value in values):
            continue
        rows.append((float(row["_step"]), [float(value) for value in values]))

    if not rows:
        return np.array([], dtype=float), np.empty((0, num_experts), dtype=float)

    rows.sort(key=lambda item: item[0])
    steps = np.array([step for step, _ in rows], dtype=float)
    usage = np.array([values for _, values in rows], dtype=float)
    return steps, usage


def _fetch_task_vector_rows(
    run,
    *,
    task_idx: int,
    num_experts: int,
    page_size: int,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    metric = _layer_vector_metric_name(task_idx, layer)
    rows = []

    for row in run.scan_history(keys=["_step", metric], page_size=page_size):
        if "_step" not in row:
            continue
        values = _coerce_usage_vector(row.get(metric), num_experts)
        if values is None:
            continue
        rows.append((float(row["_step"]), values))

    if not rows:
        return np.array([], dtype=float), np.empty((0, num_experts), dtype=float)

    rows.sort(key=lambda item: item[0])
    steps = np.array([step for step, _ in rows], dtype=float)
    usage = np.array([values for _, values in rows], dtype=float)
    return steps, usage


def _fetch_task_rows(
    run,
    *,
    task_idx: int,
    num_experts: int,
    page_size: int,
    layer: int | None,
    metric_source: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if layer is None:
        raw_steps, usage = _fetch_task_scalar_rows(
            run,
            task_idx=task_idx,
            num_experts=num_experts,
            page_size=page_size,
            layer=None,
        )
        return raw_steps, usage, "scalar"

    if metric_source in ("auto", "vector"):
        raw_steps, usage = _fetch_task_vector_rows(
            run,
            task_idx=task_idx,
            num_experts=num_experts,
            page_size=page_size,
            layer=layer,
        )
        if raw_steps.size > 0 or metric_source == "vector":
            return raw_steps, usage, "vector"

    raw_steps, usage = _fetch_task_scalar_rows(
        run,
        task_idx=task_idx,
        num_experts=num_experts,
        page_size=page_size,
        layer=layer,
    )
    return raw_steps, usage, "scalar"


def load_concatenated_usage(
    run,
    *,
    first_task: int,
    num_tasks: int,
    num_experts: int,
    x_mode: str,
    page_size: int,
    layer: int | None,
    metric_source: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, float, float]], list[str], set[str]]:
    expert_steps = [[] for _ in range(num_experts)]
    expert_usage = [[] for _ in range(num_experts)]
    task_spans = []
    warnings = []
    sources_used = set()
    next_start = 0.0

    for task_idx in range(first_task, first_task + num_tasks):
        raw_steps, usage, source = _fetch_task_rows(
            run,
            task_idx=task_idx,
            num_experts=num_experts,
            page_size=page_size,
            layer=layer,
            metric_source=metric_source,
        )
        if raw_steps.size == 0:
            if layer is None:
                warnings.append(f"No rows found for task_{task_idx:02d}.")
            else:
                warnings.append(f"No rows found for task_{task_idx:02d} layer_{layer}.")
            continue
        sources_used.add(source)

        if x_mode == "logged":
            plot_steps = raw_steps
        else:
            local_steps = raw_steps - raw_steps[0]
            plot_steps = local_steps + next_start
            next_start = float(plot_steps[-1] + _positive_step_delta(raw_steps))

        for expert_idx in range(num_experts):
            expert_steps[expert_idx].extend(plot_steps.tolist())
            expert_usage[expert_idx].extend(usage[:, expert_idx].tolist())

        task_spans.append((task_idx, float(plot_steps[0]), float(plot_steps[-1])))

    return (
        [np.array(values, dtype=float) for values in expert_steps],
        [np.array(values, dtype=float) for values in expert_usage],
        task_spans,
        warnings,
        sources_used,
    )


def _write_csv(
    output: pathlib.Path,
    expert_steps: Sequence[np.ndarray],
    expert_usage: Sequence[np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("step,expert,usage\n")
        for expert_idx, (steps, usage_values) in enumerate(zip(expert_steps, expert_usage, strict=True)):
            for step, usage in zip(steps, usage_values, strict=True):
                f.write(f"{step:g},{expert_idx},{usage:g}\n")


def plot_usage(
    expert_steps: Sequence[np.ndarray],
    expert_usage: Sequence[np.ndarray],
    task_spans: Sequence[tuple[int, float, float]],
    *,
    title: str,
    output: pathlib.Path,
    show_task_boundaries: bool,
) -> None:
    if not any(steps.size for steps in expert_steps):
        raise ValueError("No expert usage points were loaded.")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(15, 6), facecolor="white")
    colors = plt.get_cmap("tab10").colors

    for expert_idx, (steps, usage_values) in enumerate(zip(expert_steps, expert_usage, strict=True)):
        if steps.size == 0:
            continue
        ax.plot(
            steps,
            usage_values,
            label=f"Expert {expert_idx}",
            color=colors[expert_idx % len(colors)],
            linewidth=1.8,
        )

    if show_task_boundaries:
        for task_idx, start, end in task_spans:
            ax.axvline(start, color="#777777", linewidth=0.8, alpha=0.25)
            ax.text(
                (start + end) / 2,
                1.015,
                f"task_{task_idx:02d}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
        if task_spans:
            ax.axvline(task_spans[-1][2], color="#777777", linewidth=0.8, alpha=0.25)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    ax.set_xlabel("Concatenated step")
    ax.set_ylabel("Usage (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5, alpha=0.45)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run_path", help="Exact W&B run path: entity/project/run_id")
    run_group.add_argument("--run_name", help="W&B display name to search for, e.g. exp_name")
    parser.add_argument("--entity", help="W&B entity used with --run_name. Defaults to WANDB_ENTITY if set.")
    parser.add_argument("--project", default="openpi", help="W&B project for --run_name lookup. Default: openpi")
    parser.add_argument("--first_task", type=int, default=0, help="First task index. Default: 0")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of task panels to merge. Default: 10")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of expert usage lines. Default: 8")
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layer", type=int, help="Plot one MoE layer instead of global expert usage.")
    layer_group.add_argument("--layers", type=int, nargs="+", help="Plot multiple MoE layers.")
    parser.add_argument(
        "--metric_source",
        choices=("auto", "vector", "scalar"),
        default="scalar",
        help=(
            "Layer metric source. scalar uses task_XX/moe/layer_N/expert_K_usage. "
            "auto tries task_XX/moe_layers/layer_N/expert_usage first, then scalar."
        ),
    )
    parser.add_argument(
        "--x_mode",
        choices=("concat", "logged"),
        default="concat",
        help="concat rebases each task and appends it; logged uses the original W&B _step values.",
    )
    parser.add_argument("--page_size", type=int, default=1000, help="W&B history page size. Default: 1000")
    parser.add_argument("--output", type=pathlib.Path, help="Output image path, usually .png or .pdf")
    parser.add_argument(
        "--root_output_dir",
        type=pathlib.Path,
        default=pathlib.Path("figures"),
        help="Root directory for automatic outputs. Default: figures",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Directory for automatic outputs. Defaults to ROOT_OUTPUT_DIR/<wandb run display name>.",
    )
    parser.add_argument("--output_prefix", help="Optional filename prefix for automatic outputs.")
    parser.add_argument("--csv", type=pathlib.Path, help="Optional CSV export of the concatenated values")
    parser.add_argument("--csv_dir", type=pathlib.Path, help="Directory for automatic CSV exports.")
    parser.add_argument("--save_csv", action="store_true", help="Save CSV files next to the generated plots.")
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--no_task_boundaries", action="store_true", help="Hide vertical task boundary markers")
    args = parser.parse_args()

    if args.layers and (args.output is not None or args.csv is not None):
        parser.error("Use --output_dir/--csv_dir/--save_csv with --layers, not --output/--csv.")

    try:
        run = _resolve_run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    layers = args.layers if args.layers is not None else [args.layer]
    default_output_dir = args.output_dir or _default_run_output_dir(run, args.root_output_dir)

    for layer in layers:
        try:
            expert_steps, expert_usage, task_spans, warnings, sources_used = load_concatenated_usage(
                run,
                first_task=args.first_task,
                num_tasks=args.num_tasks,
                num_experts=args.num_experts,
                x_mode=args.x_mode,
                page_size=args.page_size,
                layer=layer,
                metric_source=args.metric_source,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)

        if args.layers:
            output = default_output_dir / _default_output_name(layer, args.output_prefix)
            if args.csv_dir is not None:
                csv_output = args.csv_dir / _default_csv_name(layer, args.output_prefix)
            elif args.save_csv:
                csv_output = output.with_suffix(".csv")
            else:
                csv_output = None
        else:
            output = args.output or default_output_dir / _default_output_name(layer, args.output_prefix)
            if args.csv is not None:
                csv_output = args.csv
            elif args.csv_dir is not None:
                csv_output = args.csv_dir / _default_csv_name(layer, args.output_prefix)
            elif args.save_csv:
                csv_output = output.with_suffix(".csv")
            else:
                csv_output = None

        if csv_output is not None:
            _write_csv(csv_output, expert_steps, expert_usage)
            print(f"Saved CSV to {csv_output}")

        title = args.title or f"MoE {_target_title(layer)} Expert Usage - {run.name}"
        plot_usage(
            expert_steps,
            expert_usage,
            task_spans,
            title=title,
            output=output,
            show_task_boundaries=not args.no_task_boundaries,
        )
        if layer is not None and sources_used:
            print(f"Layer {layer} metric source(s): {', '.join(sorted(sources_used))}")


if __name__ == "__main__":
    main()
