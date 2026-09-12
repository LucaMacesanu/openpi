"""Knowledge Insulation (KI) + subtask-prediction auxiliary targets for Pi0Ki.

JAX/openpi port of the lerobot patch to lerobot/policies/pi05/{modeling,
configuration,processor}_pi05.py (compute_ki_loss / compute_subtask_loss /
Pi05SubtaskLookupProcessorStep) that currently trains
yor-pi05-ablation-ki-subtask.yaml. Precomputes the two discrete-loss targets host-side
(CPU dataloader worker), fixed-shape, so they slot straight into Pi0Ki.compute_loss's
insulated/live-gradient split (see openpi.models.pi0_ki):
  - fast_action_tokens/_mask: [bos, "Action: ", <FAST-tokenized action chunk remapped
    into paligemma vocab>, "|"], via openpi's own FASTTokenizer (physical-intelligence/
    fast-style DCT+BPE tokenizer, already fit on this exact action distribution by
    nyu-finger-robot/tools/fixes/fit_fast_tokenizer.py -- same artifact the lerobot arm
    uses).
  - subtask_tokens/_mask: the active subtask's text (SARM-style VLM annotation,
    gap-backfilled -- see SubtaskLookup), tokenized with the raw PaliGemma sentencepiece
    tokenizer.

The FAST tokenizer was fit on QUANTILES-normalized action chunks (see
fit_fast_tokenizer.py), so this transform quantile-normalizes the raw action chunk
itself before tokenizing, using this arm's norm_stats q01/q99 -- independent of the
pipeline's own Normalize step (which runs later, after yor_policy.YorInputs renames
data["action"] -> "actions"; duplicating the tiny formula here avoids a
data_transforms/model_transforms ordering dependency).
"""

import bisect
import dataclasses
import functools
import json
import logging

import numpy as np
import sentencepiece

from openpi import transforms
from openpi.models import tokenizer as _tokenizer
from openpi.shared import download

logger = logging.getLogger("openpi")


@functools.lru_cache(maxsize=1)
def _load_paligemma_sentencepiece() -> sentencepiece.SentencePieceProcessor:
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


@functools.lru_cache(maxsize=4)
def _load_fast_tokenizer(fast_tokenizer_path: str, max_action_tokens: int) -> _tokenizer.FASTTokenizer:
    return _tokenizer.FASTTokenizer(max_len=max_action_tokens, fast_tokenizer_path=fast_tokenizer_path)


@functools.lru_cache(maxsize=4)
def _load_subtask_lookup(subtask_annotations_path: str) -> "SubtaskLookup":
    return SubtaskLookup(subtask_annotations_path)


class SubtaskLookup:
    """Gap-backfilled episode_index -> per-frame subtask text lookup, ported from
    lerobot's Pi05SubtaskLookupProcessorStep (lerobot/processor/tokenizer_processor.py,
    third_party/lerobot patched at $SCRATCH/lerobot-src).

    VLM subtask annotations are sparse -- they only cover the "active manipulation"
    window per subtask, leaving gaps (reach/approach at the episode start, transitions
    between subtasks, settle/idle at the end) unlabeled. A frame before the first
    subtask or between two subtasks takes the label of the NEXT subtask (the arm is
    already mid-reach for it); a frame after the LAST subtask takes RESET_LABEL (the
    episode's post-task return-to-start motion). This turns each episode's raw
    (possibly gappy) windows into a fully contiguous partition of [0, inf).

    Keyed directly on frame_index (available raw from icl-dataset's LeRobotDataset
    rows), unlike lerobot's version, which had to reconstruct it from timestamp * fps.
    """

    RESET_LABEL = "Reset to starting position."

    def __init__(self, subtask_annotations_path: str):
        windows: dict[int, tuple[list[int], list[str]]] = {}
        with open(subtask_annotations_path) as f:
            for line in f:
                record = json.loads(line)
                ep = int(record["episode_index"])
                names = record["subtask_names"]
                ends = record["subtask_end_frames"]
                # Window i starts at 0 (i=0) or subtask i's own VLM-reported start
                # (i>0). The final entry is the reset window, starting where the last
                # real subtask ends and running to episode end.
                starts = [0] + [int(s) for s in record["subtask_start_frames"][1:]] + [int(ends[-1])]
                texts = [*names, self.RESET_LABEL]
                windows[ep] = (starts, texts)
        self._windows = windows

    def __call__(self, episode_index: int, frame_index: int) -> str:
        if episode_index not in self._windows:
            raise ValueError(
                f"episode_index {episode_index} has no entry in the subtask annotations -- "
                "every training episode must be annotated for use_subtask_prediction=True."
            )
        starts, texts = self._windows[episode_index]
        return texts[bisect.bisect_right(starts, frame_index) - 1]


def _tokenize_text(text: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = _load_paligemma_sentencepiece()
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    tokens_len = len(tokens)
    if tokens_len < max_len:
        padding = [False] * (max_len - tokens_len)
        mask = [True] * tokens_len + padding
        tokens = tokens + [0] * (max_len - tokens_len)
    else:
        if tokens_len > max_len:
            logger.warning(
                f"Subtask token length ({tokens_len}) exceeds max length ({max_len}), truncating."
            )
        tokens = tokens[:max_len]
        mask = [True] * max_len
    return np.asarray(tokens), np.asarray(mask)


def _quantile_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class KiTargetInputs(transforms.DataTransformFn):
    """Adds fast_action_tokens/_mask and (optionally) subtask_tokens/_mask to the item
    dict, forwarded downstream by yor_policy.YorInputs. Must run BEFORE YorInputs in
    data_transforms.inputs (raw episode_index/frame_index/action are still present)."""

    fast_tokenizer_path: str
    max_action_tokens: int
    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]
    use_subtask_prediction: bool = True
    subtask_annotations_path: str = ""
    max_subtask_tokens: int = 32

    def __call__(self, data: dict) -> dict:
        fast_tokenizer = _load_fast_tokenizer(self.fast_tokenizer_path, self.max_action_tokens)
        raw_action = np.asarray(data["action"], dtype=np.float32)
        q01 = np.asarray(self.action_q01, dtype=np.float32)
        q99 = np.asarray(self.action_q99, dtype=np.float32)
        # _quantile_normalize matches openpi.transforms' own unclipped q01/q99 rescale
        # (fine for continuous consumers like flow-matching, which just sees an unusually
        # large float) -- but the FAST tokenizer converts each value into a discrete
        # vocab index via chr(), which hard-crashes (ValueError: chr() arg not in
        # range(0x110000)) on any true outlier action (expected for ~2% of values by
        # definition of q01/q99, and correspondingly more likely to actually occur across
        # the expanded set's much more diverse action distribution than the original
        # 4-task subset -- crashed job 16351040 at step 319/50000). Clip here, local to
        # this KI-specific consumer, rather than changing the shared unclipped helper.
        norm_action = np.clip(_quantile_normalize(raw_action, q01, q99), -1.0, 1.0)
        fast_tokens, fast_mask = fast_tokenizer.tokenize_action_only(norm_action, self.max_action_tokens)

        data = {**data, "fast_action_tokens": fast_tokens, "fast_action_tokens_mask": fast_mask}

        if self.use_subtask_prediction:
            lookup = _load_subtask_lookup(self.subtask_annotations_path)
            subtask = lookup(int(data["episode_index"]), int(data["frame_index"]))
            subtask_tokens, subtask_mask = _tokenize_text(subtask, self.max_subtask_tokens)
            data = {**data, "subtask_tokens": subtask_tokens, "subtask_tokens_mask": subtask_mask}

        return data
