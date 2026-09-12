"""Vendored (not imported) subset of viktr's retrieval math, plus a data-loading
transform that performs VICTR's vision/value nearest-neighbor retrieval (paper Sec
III-G) and builds fixed-shape context blocks for openpi.models.pi0_victr.Pi0Victr.

openpi is a separate uv workspace/venv from viktr's own venv (same reasoning as
training/config.py's LeRobotYorDataConfig precomputing its episode list to a static
JSON instead of cross-venv importing viktr.data.icl_dataset) -- these are direct
copies of the backbone-agnostic pieces of viktr/retrieval/{chunk_dictionary,
embeddings,metrics}.py, viktr/data/icl_dataset.py's RoboDopamine value-curve reader,
and viktr/policy/pi05_context.py's chunk-to-text builder, trimmed to drop anything
lerobot/torch-policy-specific (PyTorch DDP wrapping, etc). Keep in sync by hand
if the source files change.

vision_value retrieval below is a *fixed-weight* fusion (per-query z-score of
each distance array, summed) -- not the paper's learned gpsi trained jointly
with the IC-VLA via a differentiable soft top-k (viktr/retrieval/soft_topk.py's
soft_fused_retrieve, only ever used by the old lerobot/PyTorch train_victr.py
stack, and never actually trained there either). Retrieval here is a fixed,
host-side, offline precompute step for every metric including this one -- see
the "Add a vision+value fusion VICTR arm" plan for why joint/differentiable
retrieval was scoped out.

Pool files are read from scripts/convert_pool_for_openpi.py's plain-dict re-pickle
of outputs/victr/icl_pool/*.pkl, NOT the original viktr.retrieval.chunk_dictionary
pickles directly -- pickle resolves classes by their original module path, so a
pool pickled with viktr's own Chunk/ChunkDictionary classes fails to unpickle from
any environment that doesn't have `viktr` importable.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pickle
import re
from pathlib import Path
from typing import Literal

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

from openpi import transforms
from openpi.models import tokenizer as _tokenizer
from openpi.shared.image_tools import resize_with_pad_torch

# ---------------------------------------------------------------------------
# viktr/retrieval/chunk_dictionary.py
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Chunk:
    task: str
    episode_index: int
    start_frame: int
    end_frame: int
    images: dict[str, np.ndarray]  # camera_key -> (L, H, W, 3) uint8
    proprio: np.ndarray  # (L, state_dim) float32
    actions: np.ndarray  # (L, action_dim) float32
    value: np.ndarray | None = None


@dataclasses.dataclass
class ChunkDictionary:
    chunks: list[Chunk]
    key_embeddings: np.ndarray
    key_values: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)


def task_slug(task: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:80]


def load_pool(path: Path) -> ChunkDictionary:
    with open(path, "rb") as f:
        plain = pickle.load(f)
    chunks = [Chunk(**c) for c in plain["chunks"]]
    return ChunkDictionary(chunks=chunks, key_embeddings=plain["key_embeddings"], key_values=plain["key_values"])


@functools.lru_cache(maxsize=4)
def _load_pool_cached(pool_dir: str, task: str) -> ChunkDictionary:
    """Cached per-process (i.e. once per dataloader worker, not once per item).

    maxsize=4 (not unbounded): this dataset's 31 task pools total ~29GB
    (assets/victr_icl_pool_expanded_224/, images included), dominated by a few
    outliers (sort_the_items_into_their_containers.pkl alone is ~5.9GB,
    stack_the_three_cups_into_a_tower.pkl ~2.9GB). An unbounded per-worker
    cache lets each of num_workers dataloader workers independently accumulate
    every distinct task it happens to draw over an epoch -- with global
    shuffling, that's effectively the whole 29GB per worker, e.g. ~232GB
    across 8 workers, which actually OOM-killed a live-retrieval run (job
    16991490, yor_icl_fast_victr_vision_interp_expanded debug run, 2026-09-05:
    SLURM "Detected 1 oom_kill event" + a DataLoader worker killed by SIGKILL)
    -- this was flagged as an untested risk in notes/training_runs.md before
    that run finally exercised it. maxsize=4 bounds worst-case per worker to
    the 4 largest pools (~12GB) -- LRU eviction still gets full cache hits for
    the common case where nearby draws share a task (episode-contiguous
    sampling), just without retaining the entire corpus indefinitely.
    """
    return load_pool(Path(pool_dir) / f"{task_slug(task)}.pkl")


@functools.lru_cache(maxsize=None)
def _pool_max_vision_distance(pool_dir: str, task: str) -> float:
    """Per-task-pool normalization constant for action-interpolation's
    exp(-lamda * distance) weighting: max pairwise DINOv2 L2 distance within this
    task's own retrieval pool. RICL itself uses a single global constant computed
    once across its whole DROID+collected-demos corpus (assets/max_distance.json);
    scoped per-task here instead since icl-dataset's per-task pools vary widely in
    visual variety and are already loaded/cached per-task anyway (_load_pool_cached).
    Cached the same way (once per dataloader worker per task).

    Computed via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b (an N x N matrix) rather than
    the naive elementwise difference (an N x N x D tensor): this dataset's larger
    task pools have thousands of chunks at D=768, and the naive tensor OOM'd a
    48-vCPU/400G-mem node outright (e.g. ~4k chunks -> ~49GB just for that one
    temporary array) -- confirmed via a real-data smoke test of this function
    during pre-launch validation of the RICL action-interpolation port."""
    pool = _load_pool_cached(pool_dir, task)
    emb = pool.key_embeddings.astype(np.float64)
    sq_norms = np.sum(emb**2, axis=1)
    sq_dists = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * emb @ emb.T, 0.0)
    return max(float(np.sqrt(sq_dists.max())), 1e-6)


# ---------------------------------------------------------------------------
# viktr/retrieval/embeddings.py
# ---------------------------------------------------------------------------

_DINOV2_HUB_REPO = "facebookresearch/dinov2"
_DINOV2_MODEL_NAME = "dinov2_vitb14"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_dinov2_cache: torch.nn.Module | None = None


def load_dinov2() -> torch.nn.Module:
    """Loads (and caches) the frozen DINOv2 ViT-B/14 backbone, CPU-only -- this runs
    inside CPU dataloader worker processes, not competing with the GPU(s) the main JAX
    training process owns. Weights should already be cached under ~/.cache/torch/hub
    from when scripts/build_retrieval_pool.py first ran in viktr's own venv (same user,
    same home dir) -- no network fetch expected on compute nodes."""
    global _dinov2_cache
    if _dinov2_cache is None:
        model = torch.hub.load(_DINOV2_HUB_REPO, _DINOV2_MODEL_NAME)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _dinov2_cache = model
    return _dinov2_cache


def _to_dinov2_input(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


@torch.no_grad()
def embed_frames(images: np.ndarray, model: torch.nn.Module | None = None, batch_size: int = 256) -> np.ndarray:
    """CLS-token DINOv2 embedding per frame. images: (N, H, W, 3) uint8 -> (N, EMBED_DIM) float32.

    Chunks the forward pass into batch_size-sized groups -- mirrors viktr's own
    viktr.retrieval.embeddings.embed_frames exactly (same model, same math, so this
    must stay numerically equivalent, not just structurally similar). The live
    per-sample retrieval path (RetrievalContextInputs.__call__) always calls this with
    a single frame, so batch_size is a no-op there; scripts/precompute_retrieval_context.py
    calls it with a whole episode's frames at once (up to ~5,300 for the longest
    episodes), where running that as one unbatched forward pass previously spiked
    memory past 120GB and OOM'd every full-scale precompute attempt on this metric."""
    model = model if model is not None else load_dinov2()
    embeddings = []
    for start in range(0, len(images), batch_size):
        x = _to_dinov2_input(images[start : start + batch_size])
        features = model.forward_features(x)
        embeddings.append(features["x_norm_clstoken"].float().numpy())
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# viktr/retrieval/metrics.py (vision/value/vision_value fixed top-k -- vision_value
# is a fixed-weight fusion, not the paper's learned/jointly-trained gpsi; see the
# module docstring above)
# ---------------------------------------------------------------------------

RetrievalMetric = Literal["vision", "value", "vision_value"]


def _topk(scores: np.ndarray, pool: ChunkDictionary, k: int) -> list[Chunk]:
    if k > len(pool):
        raise ValueError(f"k={k} exceeds pool size {len(pool)}")
    order = np.argsort(scores)[:k]
    return [pool.chunks[i] for i in order]


def _vision_distances(query_embedding: np.ndarray, pool: ChunkDictionary) -> np.ndarray:
    """L2 distance in DINOv2 embedding space, one value per pool chunk."""
    return np.linalg.norm(pool.key_embeddings - query_embedding[None, :], axis=1)


def _value_distances(query_value: float, pool: ChunkDictionary) -> np.ndarray:
    """|progress value difference|, one value per pool chunk."""
    if pool.key_values is None:
        raise ValueError("pool has no key_values; annotate it first (viktr.value.annotate)")
    return np.abs(pool.key_values - query_value)


def vision_retrieve(query_embedding: np.ndarray, pool: ChunkDictionary, k: int) -> list[Chunk]:
    """Nearest-to-farthest, by L2 distance in DINOv2 embedding space."""
    return _topk(_vision_distances(query_embedding, pool), pool, k)


def value_retrieve(query_value: float, pool: ChunkDictionary, k: int) -> list[Chunk]:
    """Nearest-to-farthest, by |progress value difference|."""
    return _topk(_value_distances(query_value, pool), pool, k)


def fused_retrieve(
    query_embedding: np.ndarray, query_value: float, pool: ChunkDictionary, k: int
) -> list[Chunk]:
    """Nearest-to-farthest, by a fixed (non-learned) combination of vision and value
    distance: each distance array is z-scored across this query's candidate pool
    (so the two, which live on very different scales -- raw DINOv2 L2 distance vs.
    a value difference bounded in ~[0, 1] -- contribute comparably), then summed.
    Not the paper's learned gpsi (Sec III-G) -- see the module docstring."""
    d_vis = _vision_distances(query_embedding, pool)
    d_val = _value_distances(query_value, pool)
    eps = 1e-8
    z_vis = (d_vis - d_vis.mean()) / (d_vis.std() + eps)
    z_val = (d_val - d_val.mean()) / (d_val.std() + eps)
    return _topk(z_vis + z_val, pool, k)


def retrieve_chunks(
    retrieval_metric: RetrievalMetric,
    pool: ChunkDictionary,
    k: int,
    *,
    query_embedding: np.ndarray | None = None,
    query_value: float | None = None,
) -> list[Chunk]:
    """vision_retrieve/value_retrieve/fused_retrieve dispatch + nearest-to-farthest ->
    farthest-to-nearest reorder (paper Sec III-E ordering). Shared by
    RetrievalContextInputs (live, per training sample) and
    scripts/precompute_retrieval_context.py (offline, once per dataset frame)."""
    if retrieval_metric == "vision":
        chunks = vision_retrieve(query_embedding, pool, k)
    elif retrieval_metric == "value":
        chunks = value_retrieve(query_value, pool, k)
    else:
        chunks = fused_retrieve(query_embedding, query_value, pool, k)
    return list(reversed(chunks))


def build_context_block(
    chunks: list[Chunk],
    primary_camera: str,
    context_frames_per_chunk: int,
    context_text_max_length: int,
) -> dict[str, np.ndarray]:
    """Builds the fixed-shape context_images/context_image_masks/context_tokens/
    context_tokens_mask block from an already-retrieved (farthest-to-nearest) chunk
    list. Shared by RetrievalContextInputs.__call__ and the offline precompute
    script, so precomputed output is guaranteed identical to the live path's."""
    k = len(chunks)
    tokenizer = _context_tokenizer(context_text_max_length)
    frame_stack = []
    token_ids = []
    token_masks = []
    for chunk in chunks:
        frame_idx = np.linspace(0, len(chunk.proprio) - 1, context_frames_per_chunk).round().astype(np.int64)
        frame_stack.append(chunk.images[primary_camera][frame_idx])
        ids, mask = tokenizer.tokenize(chunk_to_context_text(chunk))
        token_ids.append(ids)
        token_masks.append(mask)

    frames = np.concatenate(frame_stack, axis=0)  # (k * f, H, W, 3) uint8
    resized = resize_with_pad_torch(torch.from_numpy(frames), 224, 224).numpy()
    context_images = resized.reshape(k, context_frames_per_chunk, 224, 224, 3)

    return {
        "context_images": context_images,
        "context_image_masks": np.ones((k, context_frames_per_chunk), dtype=bool),
        "context_tokens": np.stack(token_ids, axis=0),
        "context_tokens_mask": np.stack(token_masks, axis=0),
    }


# ---------------------------------------------------------------------------
# viktr/data/icl_dataset.py (RoboDopamine per-frame progress-value reader only)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _read_episode_meta(root: str) -> pd.DataFrame:
    files = sorted((Path(root) / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no meta/episodes/chunk-*/file-*.parquet under {root}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@functools.lru_cache(maxsize=None)
def _episode_lengths(root: str) -> dict[int, int]:
    eps = _read_episode_meta(root)
    return dict(zip(eps["episode_index"].astype(int), eps["length"].astype(int), strict=True))


@functools.lru_cache(maxsize=None)
def _value_estimate_index(root: str) -> dict[int, Path]:
    ve_dir = Path(root) / "meta/value_estimates"
    index: dict[int, Path] = {}
    for p in sorted(ve_dir.glob("*.json")):
        d = json.loads(p.read_text())
        ei = d.get("episode_index")
        if ei is None:
            m = re.match(r"episode_(\d+)\.json$", p.name)
            ei = int(m.group(1)) if m else None
        if ei is not None:
            index[int(ei)] = p
    return index


@functools.lru_cache(maxsize=None)
def _robodopamine_curve(root: str, episode_index: int) -> np.ndarray:
    index = _value_estimate_index(root)
    if episode_index not in index:
        raise KeyError(f"no RoboDopamine value estimate for episode_index={episode_index} under {root}")
    d = json.loads(index[episode_index].read_text())
    points = sorted(d["points"], key=lambda pt: pt["frame_index"])
    xs = np.array([pt["frame_index"] for pt in points], dtype=np.float64)
    ys = np.array([pt["progress"] for pt in points], dtype=np.float64) / 100.0
    length = _episode_lengths(root)[episode_index]
    frames = np.arange(length, dtype=np.float64)
    curve = np.interp(frames, xs, ys, left=ys[0], right=ys[-1])
    return curve.astype(np.float32)


def robodopamine_value_at(episode_index: int, frame_index: int, root: str) -> float:
    return float(_robodopamine_curve(root, episode_index)[frame_index])


# ---------------------------------------------------------------------------
# viktr/policy/pi05_context.py (chunk -> "Task/State/Action" text summary builder)
# ---------------------------------------------------------------------------

STATE_ACTION_BINS = 256  # matches PaligemmaTokenizer's own state digitization convention


def _digitize(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -1.0, 1.0)
    return np.digitize(clipped, bins=np.linspace(-1, 1, STATE_ACTION_BINS + 1)[:-1]) - 1


def chunk_to_context_text(chunk: Chunk) -> str:
    state_bins = _digitize(chunk.proprio[0])
    state_str = " ".join(map(str, state_bins))
    num_action_frames = min(len(chunk.actions), 8)
    action_idx = np.linspace(0, len(chunk.actions) - 1, num_action_frames).round().astype(np.int64)
    action_bins = _digitize(chunk.actions[action_idx])
    action_str = "; ".join(" ".join(map(str, frame)) for frame in action_bins)
    cleaned_task = chunk.task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned_task}, State: {state_str}; Action: {action_str}"


# ---------------------------------------------------------------------------
# Data-loading transform: retrieval + fixed-shape context block construction
# ---------------------------------------------------------------------------


def _parse_image(image) -> np.ndarray:
    """Same convention as openpi.policies.yor_policy._parse_image: LeRobotDataset yields
    (C, H, W) float32 in [0, 1]; retrieval wants (H, W, C) uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@functools.lru_cache(maxsize=None)
def _context_tokenizer(max_len: int) -> _tokenizer.PaligemmaTokenizer:
    return _tokenizer.PaligemmaTokenizer(max_len=max_len)


@functools.lru_cache(maxsize=None)
def _action_only_tokenizer(fast_tokenizer_path: str, max_action_tokens: int) -> _tokenizer.FASTTokenizer:
    """Same FASTTokenizer convention (and, when fast_tokenizer_path matches, the same
    fitted tokenizer) the query's own actions go through -- see
    Pi0FastVictrConfig.fast_model_tokenizer_kwargs. Used only for action-interpolation's
    nearest-chunk target, via tokenize_action_only (also used by
    openpi.policies.yor_ki.KiTargetInputs)."""
    return _tokenizer.FASTTokenizer(max_len=max_action_tokens, fast_tokenizer_path=fast_tokenizer_path)


@functools.lru_cache(maxsize=None)
def _load_precomputed_episode(
    precomputed_dir: str, metric: str, episode_index: int, include_interp: bool = False
) -> dict[str, np.ndarray]:
    """Memory-maps one episode's scripts/precompute_retrieval_context.py output --
    cached per process (i.e. once per dataloader worker per episode, not once per
    item, same convention as _load_pool_cached). include_interp=True also loads the
    action-interpolation-only arrays (exp_lamda_distance/nearest_action_tokens/
    nearest_action_tokens_mask), written only when precompute_retrieval_context.py was
    run with --use-action-interpolation."""
    metric_dir = Path(precomputed_dir) / metric
    stem = str(episode_index)
    arrays = {
        "context_images": np.load(metric_dir / f"{stem}_images.npy", mmap_mode="r"),
        "context_image_masks": np.load(metric_dir / f"{stem}_image_masks.npy", mmap_mode="r"),
        "context_tokens": np.load(metric_dir / f"{stem}_tokens.npy", mmap_mode="r"),
        "context_tokens_mask": np.load(metric_dir / f"{stem}_token_masks.npy", mmap_mode="r"),
    }
    if include_interp:
        arrays["exp_lamda_distance"] = np.load(metric_dir / f"{stem}_exp_lamda_distance.npy", mmap_mode="r")
        arrays["nearest_action_tokens"] = np.load(metric_dir / f"{stem}_nearest_action_tokens.npy", mmap_mode="r")
        arrays["nearest_action_tokens_mask"] = np.load(
            metric_dir / f"{stem}_nearest_action_tokens_mask.npy", mmap_mode="r"
        )
    return arrays


@dataclasses.dataclass(frozen=True)
class RetrievalContextInputs(transforms.DataTransformFn):
    """Must run BEFORE openpi.policies.yor_policy.YorInputs in data_transforms.inputs:
    needs the item's raw episode_index/frame_index/prompt/primary-camera image, which
    YorInputs replaces with pi05's image-slot keys. Performs VICTR's hard top-k
    retrieval against a precomputed ChunkDictionary pool and adds fixed-shape
    context_images/context_tokens fields the rest of the pipeline treats like any other
    batch field -- k (num_context_chunks) and context_frames_per_chunk are both static
    per-config, so every item produces identically-shaped output, fitting jax.jit
    without any dynamic shapes. Retrieval itself (DINOv2 embedding + nearest-neighbor
    lookup) is a non-jittable host-side computation, so it happens here rather than
    inside the model's compute_loss."""

    pool_dir: str
    icl_dataset_root: str
    primary_camera: str
    retrieval_metric: RetrievalMetric
    num_context_chunks: int
    context_frames_per_chunk: int
    context_text_max_length: int
    # When set, reads scripts/precompute_retrieval_context.py's output instead of
    # computing retrieval live -- an O(1) per-episode array lookup replacing the
    # DINOv2 embed + pool search + tokenization that otherwise runs on every sample
    # every step. None (default) preserves the exact live behavior below.
    precomputed_dir: str | None = None

    # RICL-style action interpolation (notes/action_interpolation.md,
    # openpi.models.pi0_fast_victr.Pi0FastVictr.interpolate_actions). retrieval_metric
    # must be "vision" (RICL's own mechanism is DINOv2-distance-based; there's no
    # "value distance" analog). Works with either live retrieval (precomputed_dir=None)
    # or a precomputed_dir written by precompute_retrieval_context.py
    # --use-action-interpolation (which additionally precomputes the top-1 vision
    # distance and nearest-chunk action tokens this branch needs -- see
    # _load_precomputed_episode's include_interp). action_horizon must be set to the
    # model's own action_horizon: the nearest chunk's raw (context_chunk_size,
    # action_dim) actions are resampled (linear interpolation over the chunk's own
    # time axis) to this length before FAST-tokenizing, since the pool's chunks are
    # built at context_chunk_size (e.g. 10 frames) independent of action_horizon (e.g.
    # 30) -- RICL itself retrieves a native action_horizon-length window directly (no
    # chunking, no resampling); doing the same here would need a pool rebuilt at
    # chunk_size=action_horizon, which this takes as a documented approximation to
    # avoid.
    use_action_interpolation: bool = False
    lamda: float = 10.0
    max_action_tokens: int = 192
    fast_tokenizer_path: str | None = None
    action_horizon: int | None = None
    # Mirrors LeRobotYorDataConfig.drop_base_lift / yor_policy.YorInputs' own action
    # dim-16:20 handling (base_vel + lift_cmd), applied to the *neighbor's* pool
    # actions before normalizing/tokenizing them below -- so the neighbor's action
    # vector has the exact same shape/convention as the query's own (YorInputs
    # already applies this to `data["action"]`, downstream of this transform). Only
    # meaningful when canonical_actions=False (see below) -- the OLD raw 20-dim pool
    # representation (quat+xyz+grip+base_vel+lift_cmd) is what this dim-16:20 slicing
    # was written for.
    drop_base_lift: bool = False
    # When True, the pool's own `actions` arrays are already in the ICRA canonical
    # 20-dim encoding (delta-pos + rot6d + gripper per arm, notes/ICRA_plan.md Sec 1 --
    # see the pool patch in that plan's Sec 1a addendum), which has NO base_vel/
    # lift_cmd dims at all -- dims 16:20 of a canonical action are the tail of the
    # RIGHT arm's rot6d plus its gripper, not base/lift. Zeroing or dropping them the
    # way drop_base_lift does for the old raw representation would corrupt the
    # neighbor's action vector, not sanitize it. Set True (drop_base_lift ignored) by
    # every canonical DataConfig (LeRobotYorVictrCanonicalDataConfig) that turns on
    # use_action_interpolation.
    canonical_actions: bool = False
    # Quantile norm_stats for the "actions" key (LeRobotYorVictrDataConfig.create()
    # passes self.create_base_config(...).norm_stats["actions"]) -- needed to put the
    # neighbor's actions on the SAME normalized scale as the query's own before
    # FAST-tokenizing them (see the interpolation branch below for why this matters).
    action_norm_stats: transforms.NormStats | None = None
    # Inverse of DataConfig.task_prompt_overrides (overridden display prompt -> raw
    # lerobot task string), passed in by LeRobotYorVictrDataConfig.create(). Pool
    # files (assets/victr_icl_pool*/*.pkl) are named by task_slug() of the RAW task
    # string (built before any override existed), but by the time this transform
    # runs, data_loader.py's PromptFromLeRobotTask has already replaced data["prompt"]
    # with the overridden display prompt -- every one of YOR_EXPANDED_TASK_PROMPT_
    # OVERRIDES' 8 entries maps to a different string than its key, so looking up the
    # pool by data["prompt"] directly slug-mismatches for any overridden task (only
    # ever exercised by the live-retrieval path, precomputed_dir=None -- every
    # precomputed-context config takes the episode-index array-lookup branch above
    # and never hits this). Confirmed via job 16520341's
    # FileNotFoundError-equivalent crash (missing .pkl for the overridden slug).
    prompt_to_task_overrides: dict[str, str] | None = None

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if self.precomputed_dir is not None:
            episode_arrays = _load_precomputed_episode(
                self.precomputed_dir,
                self.retrieval_metric,
                int(data["episode_index"]),
                include_interp=self.use_action_interpolation,
            )
            frame_index = int(data["frame_index"])
            context = {key: np.array(arr[frame_index]) for key, arr in episode_arrays.items()}
            return {**data, **context}

        task = str(data["prompt"])
        if self.prompt_to_task_overrides is not None:
            task = self.prompt_to_task_overrides.get(task, task)
        pool = _load_pool_cached(self.pool_dir, task)
        k = self.num_context_chunks

        query_embedding = None
        query_value = None
        if self.retrieval_metric in ("vision", "vision_value"):
            query_frame = _parse_image(data[self.primary_camera])
            query_embedding = embed_frames(query_frame[None])[0]
        if self.retrieval_metric in ("value", "vision_value"):
            query_value = robodopamine_value_at(
                int(data["episode_index"]), int(data["frame_index"]), self.icl_dataset_root
            )
        chunks = retrieve_chunks(
            self.retrieval_metric, pool, k, query_embedding=query_embedding, query_value=query_value
        )

        context = build_context_block(
            chunks, self.primary_camera, self.context_frames_per_chunk, self.context_text_max_length
        )

        if self.use_action_interpolation:
            if self.retrieval_metric != "vision":
                raise ValueError("use_action_interpolation requires retrieval_metric='vision'")
            context.update(
                action_interpolation_extras(
                    chunks=chunks,
                    query_embedding=query_embedding,
                    pool=pool,
                    pool_dir=self.pool_dir,
                    task=task,
                    lamda=self.lamda,
                    max_action_tokens=self.max_action_tokens,
                    fast_tokenizer_path=self.fast_tokenizer_path,
                    action_horizon=self.action_horizon,
                    drop_base_lift=self.drop_base_lift,
                    canonical_actions=self.canonical_actions,
                    action_norm_stats=self.action_norm_stats,
                )
            )

        return {**data, **context}


def action_interpolation_extras(
    *,
    chunks: list[Chunk],
    query_embedding: np.ndarray,
    pool: ChunkDictionary,
    pool_dir: str,
    task: str,
    lamda: float,
    max_action_tokens: int,
    fast_tokenizer_path: str,
    action_horizon: int,
    drop_base_lift: bool,
    canonical_actions: bool,
    action_norm_stats: transforms.NormStats | None,
) -> dict[str, np.ndarray]:
    """RICL-style action-interpolation extras (exp_lamda_distance, nearest_action_tokens,
    nearest_action_tokens_mask) for the top-1 retrieved chunk. Factored out of
    RetrievalContextInputs.__call__'s live path so scripts/precompute_retrieval_context.py
    --use-action-interpolation can produce byte-identical output offline -- same
    reasoning as build_context_block being shared between the two."""
    # chunks is farthest-to-nearest (retrieve_chunks reverses vision_retrieve's
    # nearest-to-farthest order for context-prepending) -- the nearest is last.
    nearest = chunks[-1]
    actions = nearest.actions.astype(np.float32, copy=True)
    if canonical_actions:
        # Already the ICRA canonical 20-dim encoding (delta-pos + rot6d + gripper per
        # arm) -- no base_vel/lift_cmd dims to zero or drop; see
        # RetrievalContextInputs.canonical_actions' docstring.
        pass
    elif drop_base_lift:
        # Same dim-16:20 (base_vel + lift_cmd) handling YorInputs applies to the
        # query's own actions -- keeps the neighbor's action vector the same shape
        # and convention the query's postfix was FAST-tokenized with.
        actions = actions[..., :16]
    else:
        actions[..., 16:20] = 0.0
    if actions.shape[0] != action_horizon:
        src_t = np.linspace(0.0, 1.0, actions.shape[0])
        dst_t = np.linspace(0.0, 1.0, action_horizon)
        actions = np.stack(
            [np.interp(dst_t, src_t, actions[:, d]) for d in range(actions.shape[1])], axis=1
        ).astype(np.float32)

    # Quantile-normalize to match what the query's own actions go through
    # (openpi.transforms.Normalize._normalize_quantile, applied to `data["action"]`
    # downstream of this transform) before FAST-tokenizing below. Without this,
    # the neighbor's actions get tokenized straight from raw physical units
    # (meters/radians) while the query's postfix tokens encode quantile-
    # normalized ([-1, 1]-ish) values -- two different numeric domains sharing
    # the same FAST vocabulary by coincidence only, so `nearest_action_tokens`
    # would almost never actually match the query's own target tokens and the
    # interpolation weight would be blending toward a essentially-arbitrary
    # token id. This was silently wrong (no crash) until traced by hand -- not
    # caught by the earlier debug runs, which never got far enough to reveal
    # it (see yor_retrieval._load_pool_cached's docstring for the OOM that was
    # blocking those).
    if action_norm_stats is not None:
        q01 = np.asarray(action_norm_stats.q01)[..., : actions.shape[-1]]
        q99 = np.asarray(action_norm_stats.q99)[..., : actions.shape[-1]]
        actions = (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    top1_distance = float(_vision_distances(query_embedding, pool).min())
    max_distance = _pool_max_vision_distance(pool_dir, task)
    normalized = np.clip(top1_distance / max_distance, 0.0, 1.0)
    exp_lamda_distance = np.array(np.exp(-lamda * normalized), dtype=np.float32)

    fast_tok = _action_only_tokenizer(fast_tokenizer_path, max_action_tokens)
    tokens, mask = fast_tok.tokenize_action_only(actions, max_action_tokens)
    # Drop tokenize_action_only's own leading bos: the query's postfix (which
    # this is blended against, at Pi0FastVictr.compute_loss/sample_actions)
    # starts directly with "Action: ..." -- no bos of its own mid-sequence.
    return {
        "exp_lamda_distance": exp_lamda_distance,
        "nearest_action_tokens": tokens[1:],
        "nearest_action_tokens_mask": mask[1:],
    }
