"""Offline precompute of VICTR retrieval context (paper Sec III-E) for every frame of
every requested episode, so a fast training-time lookup (RetrievalContextInputs'
precomputed_dir path) can replace the live per-sample DINOv2 embed + pool search +
tokenization with an O(1) array index -- see the "Precompute VICTR retrieval context
for the full icl-dataset" plan for the motivation (training GPU utilization).

Must run under openpi's own venv (not viktr's): reuses openpi.policies.yor_retrieval
verbatim (retrieve_chunks, build_context_block, embed_frames, robodopamine_value_at,
load_pool) so precomputed output is guaranteed identical to what
RetrievalContextInputs.__call__ produces live, not a reimplementation. Pool loading
itself uses this script's own _load_pool_bounded (maxsize=1), not yor_retrieval's
_load_pool_cached (maxsize=None, correct for the live path's shuffled task order but
what actually caused this script's worst OOMs at the expanded task set's pool sizes
-- see that function's docstring below).

The task -> episode-indices mapping is read from a static JSON
(outputs/victr/icl_pool*/all_episodes.json, written by scripts/build_retrieval_pool.py)
rather than importing viktr.data.icl_dataset directly -- same cross-venv-via-static-
JSON convention openpi/training/config.py's LeRobotYorDataConfig already uses.

--pool-dir should point at a resize_pool_images.py output (224x224 chunk images), not
build_retrieval_pool.py's raw 480x640 output directly -- see that script's docstring;
running against un-resized pools risks the same OOM this task-based sharding alone
doesn't fully solve for (see _assign_tasks_to_shards' docstring).

Usage (one shard of a job array; --num-shards/--shard-index split TASKS (not
episodes) into independent, restartable chunks, bin-packed by pool size via
_assign_tasks_to_shards -- each shard skips episodes whose output already exists):

    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_retrieval_context.py \
        --pool-dir third_party/openpi/assets/victr_icl_pool_expanded_224 \
        --all-episodes-json outputs/victr/icl_pool_expanded/all_episodes.json \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
        --out-dir outputs/victr/retrieval_context_expanded \
        --metric vision \
        --num-shards 16 --shard-index 0

Output is pinned to the num_context_chunks/context_frames_per_chunk/
context_text_max_length values passed on the CLI (defaults match the current
Pi0VictrConfig defaults: 1/1/64) -- if those model-config values change, precompute
must be rerun.

Each episode is processed in --frame-window-size-sized windows (default 256, matching
embed_frames' own DINOv2 batch size) rather than decoding+embedding the whole episode
up front: for metric=vision this bounds peak memory regardless of episode length (some
episodes run up to ~5,300+ frames), and for both metrics it means a killed/timed-out
job (wall-time limit, preemption) resumes the episode from its last completed window
on the next run instead of redoing it from frame 0 -- see precompute_episode's
docstring. Per-episode progress lives in <out_dir>/<metric>/<episode>_progress.json
plus <episode>_{images,image_masks,tokens,token_masks}.npy.partial next to the final
output; these are cleaned up automatically once an episode finishes (renamed to their
canonical non-.partial names) and are safe to delete by hand to force a from-scratch
redo of one episode.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from openpi.policies.yor_retrieval import (  # noqa: E402
    ChunkDictionary,
    _parse_image,
    action_interpolation_extras,
    build_context_block,
    embed_frames,
    load_pool,
    retrieve_chunks,
    robodopamine_value_at,
    task_slug,
)
from openpi.shared.normalize import NormStats  # noqa: E402


@functools.lru_cache(maxsize=1)
def _load_pool_bounded(pool_dir: str, task: str) -> ChunkDictionary:
    """Own bounded cache for this script, NOT yor_retrieval._load_pool_cached's
    (maxsize=None) one -- the expanded task set's per-task pools are up to 36GB each
    (176GB total across all 31 tasks; the original 4-task pools topped out at 4.9GB),
    so an unbounded cache that accumulates every task a shard happens to touch over
    its lifetime is what actually caused job 16350893's 32/32-shard OOM, not
    embed_frames or the raw-frame loading fixed above. maxsize=1 is free here (not
    just safe) because `pairs` below is built task-major then strided, so a shard's
    (task, episode) pairs already arrive in task-contiguous runs -- this never
    reloads a pool it wouldn't have had to load anyway. Left as a separate cache
    (rather than shrinking the shared one) because the live training path calls
    _load_pool_cached with shuffled, non-contiguous task order, where a maxsize=1
    cache would thrash on every sample."""
    return load_pool(Path(pool_dir) / f"{task_slug(task)}.pkl")


def _assign_tasks_to_shards(tasks: list[str], pool_dir: Path, num_shards: int) -> list[list[str]]:
    """Greedy longest-processing-time-first bin-packing of tasks across shards, weighted
    by each task's pool file size (a good proxy for its total precompute workload --
    scales with total chunks/frames for that task). Each task is assigned to exactly
    ONE shard -- unlike the old flat (task, episode) interleave-then-stride scheme,
    this guarantees a task's (possibly huge, up to 36GB) pool is only ever loaded by
    one process, never redundantly copied into many shards' memory at once, which was
    the real cause of job 16350893's OOM (see this module's docstring). Deterministic:
    every shard re-derives the identical assignment from the same inputs, so no shared
    state needs to be written between shards."""
    sizes = [(pool_dir / f"{task_slug(t)}.pkl").stat().st_size for t in tasks]
    order = sorted(range(len(tasks)), key=lambda i: sizes[i], reverse=True)
    bins: list[list[str]] = [[] for _ in range(num_shards)]
    bin_totals = [0] * num_shards
    for i in order:
        j = min(range(num_shards), key=lambda b: bin_totals[b])
        bins[j].append(tasks[i])
        bin_totals[j] += sizes[i]
    return bins


_INTERP_ARRAY_KEYS = ("exp_lamda_distance", "nearest_action_tokens", "nearest_action_tokens_mask")


def _episode_output_paths(out_dir: Path, metric: str, episode_index: int, *, interp: bool = False) -> dict[str, Path]:
    metric_dir = out_dir / metric
    stem = str(episode_index)
    paths = {
        "images": metric_dir / f"{stem}_images.npy",
        "image_masks": metric_dir / f"{stem}_image_masks.npy",
        "tokens": metric_dir / f"{stem}_tokens.npy",
        "token_masks": metric_dir / f"{stem}_token_masks.npy",
    }
    if interp:
        for key in _INTERP_ARRAY_KEYS:
            paths[key] = metric_dir / f"{stem}_{key}.npy"
    return paths


def _already_done(paths: dict[str, Path]) -> bool:
    return all(p.exists() for p in paths.values())


def _partial_paths(out_dir: Path, metric: str, episode_index: int, *, interp: bool = False) -> dict[str, Path]:
    metric_dir = out_dir / metric
    stem = str(episode_index)
    paths = {
        "images": metric_dir / f"{stem}_images.npy.partial",
        "image_masks": metric_dir / f"{stem}_image_masks.npy.partial",
        "tokens": metric_dir / f"{stem}_tokens.npy.partial",
        "token_masks": metric_dir / f"{stem}_token_masks.npy.partial",
        "progress": metric_dir / f"{stem}_progress.json",
    }
    if interp:
        for key in _INTERP_ARRAY_KEYS:
            paths[key] = metric_dir / f"{stem}_{key}.npy.partial"
    return paths


@dataclasses.dataclass(frozen=True)
class InterpConfig:
    """RICL action-interpolation precompute params -- mirrors the
    RetrievalContextInputs fields action_interpolation_extras() takes, minus the
    per-frame chunks/query_embedding/pool/task args this module supplies itself.
    metric must be "vision" (asserted in main()) -- action interpolation is
    DINOv2-distance-based only."""

    lamda: float
    max_action_tokens: int
    fast_tokenizer_path: str
    action_horizon: int
    drop_base_lift: bool
    canonical_actions: bool
    action_norm_stats: NormStats | None


def precompute_episode(
    *,
    task: str,
    episode_index: int,
    metric: str,
    pool_dir: str,
    icl_dataset_root: str,
    out_dir: Path,
    primary_camera: str,
    k: int,
    context_frames_per_chunk: int,
    context_text_max_length: int,
    frame_window_size: int,
    interp: InterpConfig | None = None,
) -> int:
    """Processes one episode's frames in fixed-size windows (decode -> embed -> retrieve
    -> build_context_block), writing each window's output directly into on-disk
    per-array memmaps and recording how many frames are done in a JSON sidecar --
    instead of the old scheme of decoding+embedding every frame of the episode into one
    big in-memory array and only ever writing output once the whole episode finished.
    That meant a killed/timed-out job (e.g. job 16357824, TIMEOUT mid-episode on a
    long-episode task) silently threw away 100% of an unfinished episode's work on
    every retry. Here, a resumed run reopens the existing .partial memmaps and
    continues from the last completed window, so repeated re-submissions make forward
    progress on a straggler episode instead of restarting it from frame 0 each time.
    Also bounds peak memory to frame_window_size frames regardless of episode length
    (some episodes run up to ~5,300+ frames) -- the original OOM cause for this metric
    before embed_frames() batched its own forward pass; that batching alone didn't
    help the *decode* step, which this fixes too.

    Returns the number of frames newly processed this call (0 if the episode was
    already fully done, in which case the caller should have already skipped it via
    _already_done on the final paths -- this is just a safety net).
    """
    final_paths = _episode_output_paths(out_dir, metric, episode_index, interp=interp is not None)
    if _already_done(final_paths):
        return 0

    pool = _load_pool_bounded(pool_dir, task)
    # See viktr.data.icl_dataset.load_episode_arrays: a few icl-dataset episodes sit
    # right on lerobot's default tolerance_s=1e-4 boundary (float-rounding noise),
    # which raises FrameTimestampError under the default -- doubled here to match.
    dataset = LeRobotDataset("icl-dataset", root=icl_dataset_root, episodes=[episode_index], tolerance_s=2e-4)
    num_frames = len(dataset)

    partial_paths = _partial_paths(out_dir, metric, episode_index, interp=interp is not None)
    partial_paths["progress"].parent.mkdir(parents=True, exist_ok=True)
    frames_done = (
        json.loads(partial_paths["progress"].read_text())["frames_done"]
        if partial_paths["progress"].exists()
        else 0
    )

    array_keys = ("images", "image_masks", "tokens", "token_masks")
    block_keys = ("context_images", "context_image_masks", "context_tokens", "context_tokens_mask")
    if interp is not None:
        array_keys = array_keys + _INTERP_ARRAY_KEYS
        block_keys = block_keys + _INTERP_ARRAY_KEYS  # action_interpolation_extras' own key names, no prefix

    memmaps: dict[str, np.memmap] | None = None
    if frames_done > 0:
        memmaps = {key: np.lib.format.open_memmap(partial_paths[key], mode="r+") for key in array_keys}

    for window_start in range(frames_done, num_frames, frame_window_size):
        window_end = min(window_start + frame_window_size, num_frames)

        if metric in ("vision", "vision_value"):
            # Only load+stack this window's raw frames when actually needed for DINOv2
            # embedding -- the value-only metric (robodopamine_value_at, disk-backed,
            # no video frames needed) never touches raw frames at all.
            window_frames = np.stack(
                [_parse_image(dataset[i][primary_camera]) for i in range(window_start, window_end)], axis=0
            )  # (w, H, W, 3) uint8
            window_embeddings = embed_frames(window_frames)  # (w, EMBED_DIM)

        window_blocks = []
        for offset, frame_index in enumerate(range(window_start, window_end)):
            query_embedding = window_embeddings[offset] if metric in ("vision", "vision_value") else None
            query_value = (
                robodopamine_value_at(episode_index, frame_index, icl_dataset_root)
                if metric in ("value", "vision_value")
                else None
            )
            chunks = retrieve_chunks(metric, pool, k, query_embedding=query_embedding, query_value=query_value)
            block = build_context_block(chunks, primary_camera, context_frames_per_chunk, context_text_max_length)
            if interp is not None:
                block.update(
                    action_interpolation_extras(
                        chunks=chunks,
                        query_embedding=query_embedding,
                        pool=pool,
                        pool_dir=pool_dir,
                        task=task,
                        lamda=interp.lamda,
                        max_action_tokens=interp.max_action_tokens,
                        fast_tokenizer_path=interp.fast_tokenizer_path,
                        action_horizon=interp.action_horizon,
                        drop_base_lift=interp.drop_base_lift,
                        canonical_actions=interp.canonical_actions,
                        action_norm_stats=interp.action_norm_stats,
                    )
                )
            window_blocks.append(block)

        if memmaps is None:
            # First window of a from-scratch episode -- learn each array's exact
            # per-frame shape/dtype from the block actually produced, rather than
            # hardcoding it, so this stays correct if build_context_block's output
            # ever changes.
            sample = window_blocks[0]
            memmaps = {
                array_key: np.lib.format.open_memmap(
                    partial_paths[array_key],
                    mode="w+",
                    dtype=np.asarray(sample[block_key]).dtype,
                    shape=(num_frames, *np.asarray(sample[block_key]).shape),
                )
                for array_key, block_key in zip(array_keys, block_keys, strict=True)
            }
        elif window_start == frames_done:
            # Resuming into an existing .partial episode -- make sure this run's CLI
            # config (k / context_frames_per_chunk / context_text_max_length /
            # --use-action-interpolation) still matches what the partial files were
            # started with.
            sample = window_blocks[0]
            for array_key, block_key in zip(array_keys, block_keys, strict=True):
                expected = np.asarray(sample[block_key]).shape
                actual = memmaps[array_key].shape[1:]
                if actual != expected:
                    raise ValueError(
                        f"episode {episode_index}: existing {partial_paths[array_key]} has per-frame shape "
                        f"{actual}, but this run's --num-context-chunks/--context-frames-per-chunk/"
                        f"--context-text-max-length/--use-action-interpolation produce {expected} -- delete the "
                        f"stale .partial and _progress.json files for this episode and rerun."
                    )

        for offset, frame_index in enumerate(range(window_start, window_end)):
            b = window_blocks[offset]
            for array_key, block_key in zip(array_keys, block_keys, strict=True):
                memmaps[array_key][frame_index] = b[block_key]

        for m in memmaps.values():
            m.flush()
        partial_paths["progress"].write_text(json.dumps({"frames_done": window_end, "total_frames": num_frames}))
        print(f"    window {window_start}-{window_end - 1}/{num_frames} done", flush=True)

    del memmaps  # release memmap handles before rename (Windows needs this; harmless elsewhere)
    for key, final_path in final_paths.items():
        partial_paths[key].rename(final_path)
    partial_paths["progress"].unlink()

    return num_frames - frames_done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--all-episodes-json", required=True, type=Path)
    parser.add_argument("--icl-dataset-root", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--primary-camera", default="observation.images.zed")
    parser.add_argument("--metric", choices=["vision", "value", "vision_value"], required=True)
    parser.add_argument("--num-context-chunks", type=int, default=1)
    parser.add_argument("--context-frames-per-chunk", type=int, default=1)
    parser.add_argument("--context-text-max-length", type=int, default=64)
    parser.add_argument("--tasks", default="all", help="'all' or a comma-separated list of task strings")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N (task, episode) pairs in this shard -- for pilot timing runs")
    parser.add_argument(
        "--frame-window-size",
        type=int,
        default=256,
        help="decode/embed/build-context in windows of this many frames at a time (matches embed_frames' own "
        "DINOv2 batch size by default), writing each window to disk before moving on -- bounds peak memory and "
        "lets a killed/timed-out job resume a straggler episode mid-way instead of redoing it from frame 0",
    )
    parser.add_argument(
        "--use-action-interpolation",
        action="store_true",
        help="also precompute exp_lamda_distance/nearest_action_tokens/nearest_action_tokens_mask "
        "(yor_retrieval.action_interpolation_extras) per frame, so RetrievalContextInputs with "
        "use_action_interpolation=True can read this precomputed_dir instead of live retrieval. "
        "Requires --metric vision.",
    )
    parser.add_argument("--lamda", type=float, default=10.0)
    parser.add_argument("--max-action-tokens", type=int, default=192)
    parser.add_argument("--fast-tokenizer-path", type=str, default=None)
    parser.add_argument(
        "--action-horizon", type=int, default=None, help="model's action_horizon -- required with --use-action-interpolation"
    )
    parser.add_argument(
        "--drop-base-lift",
        action="store_true",
        help="OLD raw 20-dim pool representation only (quat+xyz+grip+base_vel+lift_cmd) -- mutually exclusive "
        "with --canonical-actions; ignored if --canonical-actions is set",
    )
    parser.add_argument(
        "--canonical-actions",
        action="store_true",
        help="pool's `actions` are already the ICRA canonical 20-dim encoding (delta-pos+rot6d+gripper per arm, "
        "no base/lift dims) -- see yor_retrieval.RetrievalContextInputs.canonical_actions' docstring. Set this for "
        "any victr_icl_pool_canonical_*/ pool.",
    )
    parser.add_argument(
        "--action-norm-stats-json",
        type=Path,
        default=None,
        help="assets/<config>/icl-dataset/norm_stats.json to read the 'actions' q01/q99 from, for quantile-"
        "normalizing the neighbor's actions before FAST-tokenizing -- required with --use-action-interpolation",
    )
    args = parser.parse_args()

    interp: InterpConfig | None = None
    if args.use_action_interpolation:
        if args.metric != "vision":
            raise ValueError("--use-action-interpolation requires --metric vision")
        if args.action_horizon is None:
            raise ValueError("--use-action-interpolation requires --action-horizon")
        if not args.fast_tokenizer_path:
            raise ValueError("--use-action-interpolation requires --fast-tokenizer-path")
        action_norm_stats = None
        if args.action_norm_stats_json is not None:
            stats = json.loads(args.action_norm_stats_json.read_text())["norm_stats"]["actions"]
            action_norm_stats = NormStats(**stats)
        interp = InterpConfig(
            lamda=args.lamda,
            max_action_tokens=args.max_action_tokens,
            fast_tokenizer_path=args.fast_tokenizer_path,
            action_horizon=args.action_horizon,
            drop_base_lift=args.drop_base_lift,
            canonical_actions=args.canonical_actions,
            action_norm_stats=action_norm_stats,
        )

    all_episodes: dict[str, list[int]] = json.loads(args.all_episodes_json.read_text())
    tasks = list(all_episodes) if args.tasks == "all" else args.tasks.split(",")

    shard_tasks = _assign_tasks_to_shards(tasks, args.pool_dir, args.num_shards)[args.shard_index]
    pairs = [(task, ep) for task in shard_tasks for ep in all_episodes[task]]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(shard_tasks)} tasks "
        f"({', '.join(shard_tasks) or 'none'}), {len(pairs)} (task, episode) pairs to process"
    )

    metric_dir = args.out_dir / args.metric
    metric_dir.mkdir(parents=True, exist_ok=True)

    for i, (task, episode_index) in enumerate(pairs):
        paths = _episode_output_paths(args.out_dir, args.metric, episode_index, interp=interp is not None)
        if _already_done(paths):
            print(f"[{i + 1}/{len(pairs)}] episode {episode_index}: already done, skipping")
            continue

        start = time.time()
        n_new = precompute_episode(
            task=task,
            episode_index=episode_index,
            metric=args.metric,
            pool_dir=str(args.pool_dir),
            icl_dataset_root=args.icl_dataset_root,
            out_dir=args.out_dir,
            primary_camera=args.primary_camera,
            k=args.num_context_chunks,
            context_frames_per_chunk=args.context_frames_per_chunk,
            context_text_max_length=args.context_text_max_length,
            frame_window_size=args.frame_window_size,
            interp=interp,
        )
        elapsed = time.time() - start
        rate = f" ({elapsed / n_new:.3f}s/frame)" if n_new else ""
        print(f"[{i + 1}/{len(pairs)}] episode {episode_index} ({task!r}, {n_new} new frames): {elapsed:.1f}s{rate}")


if __name__ == "__main__":
    main()
