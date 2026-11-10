"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_victr as pi0_victr
import openpi.models.pi0_fast_victr as pi0_fast_victr
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.yor_policy as yor_policy
import openpi.policies.yor_retrieval as yor_retrieval
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Local dataset root (bypasses HF hub resolution/download). If None, the dataset is
    # resolved from the HF hub cache as usual.
    root: str | None = None
    # Episode indices to restrict the dataset to. If None, all episodes are used.
    episodes: Sequence[int] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Corrections applied to the raw LeRobot dataset task string before it becomes
    # `prompt` (only takes effect when prompt_from_task=True): raw on-disk task
    # string -> corrected prompt text. Empty by default (no effect on any config
    # that doesn't set it). For a mislabeled/underspecified task string that can't
    # be fixed by mutating the dataset files themselves, this is the hook to fix
    # what the policy is actually trained on.
    task_prompt_overrides: dict[str, str] = dataclasses.field(default_factory=dict)

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # Forwarded to TokenizePrompt (PI05 only -- the only model type whose prompt
    # embeds state at all, via discrete_state_input). See
    # _transforms.TokenizePrompt.state_dropout_prob's docstring.
    state_dropout_prob: float = 0.0

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            state_dropout_prob=self.state_dropout_prob,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# --- VICTR release paths ------------------------------------------------------------
# Every VICTR/yor_icl_* config below reads its on-disk inputs from these. Defaults
# resolve inside this openpi checkout (the layout the release's download/prepare
# scripts produce); override any of them with the matching environment variable.
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _victr_path(env_var: str, default: str) -> str:
    return os.environ.get(env_var, str(_OPENPI_ROOT / default))


# LeRobot-format training dataset (icl-dataset-fixed-obs).
VICTR_DATA_ROOT = _victr_path("VICTR_DATA_ROOT", "data/icl-dataset-fixed-obs")
# Fitted FAST action tokenizer for the canonical action space.
VICTR_FAST_TOKENIZER = _victr_path("VICTR_FAST_TOKENIZER", "assets/fast_tokenizer/yor-icl-canonical")
# Output root of scripts/precompute_retrieval_context.py (one subdir per pool variant).
VICTR_RETRIEVAL_CONTEXT_DIR = _victr_path("VICTR_RETRIEVAL_CONTEXT_DIR", "assets/retrieval_context")
# Stage-1 checkpoint (yor_icl_pi05_canonical_extended, step 40k) the ctx3cam VICTR arms
# and the warm-start baseline initialize from.
VICTR_WARMSTART_PARAMS = _victr_path(
    "VICTR_WARMSTART_PARAMS", "checkpoints/yor_icl_pi05_canonical_extended/stage1/40000/params"
)


# The icl-dataset Hub repo (adityx23/icl-dataset, commit 5948ba4) fixed a mislabel:
# the "cluttered table" orange-cube episodes' actual destination is the box, not the
# plate. This project's on-disk icl-dataset copy isn't mutated to match (see
# viktr.data.icl_dataset.EXPANDED_PROMPT_OVERRIDES, which corrects the same string
# for the 31-task expanded set) -- this is the openpi-side equivalent, applied via
# DataConfig.task_prompt_overrides so the pi05/KI/VICTR 4-task-pnp configs below
# (all of which include this task) train on the corrected prompt.
YOR_TASK_PROMPT_OVERRIDES: dict[str, str] = {
    "pick up the orange cube and place it on the plate with the left arm on a cluttered table": (
        "pick up the orange cube and place it in the box with the left arm"
    ),
}

# Full port of viktr.data.icl_dataset.EXPANDED_PROMPT_OVERRIDES (see that dict's
# comment for the per-entry rationale) for the 31-task expanded-set configs below --
# unlike YOR_TASK_PROMPT_OVERRIDES above (scoped to the 4-task pnp subset, where only
# the cluttered-table entry is relevant), the expanded set includes every task these
# overrides touch.
YOR_EXPANDED_TASK_PROMPT_OVERRIDES: dict[str, str] = {
    "pick up the orange cube and place it on the plate with the left arm on a cluttered table": (
        "pick up the orange cube and place it in the box with the left arm"
    ),
    "pick up the orange cube and place it on the plate with the left arm on an uncluttered table": (
        "pick up the orange cube and place it on the plate with the left arm"
    ),
    "pick up the orange cube and place it on the plate with the right arm on a cluttered table": (
        "pick up the orange cube and place it in the box with the right arm"
    ),
    "pick up the orange cube and place it on the plate with the right arm on an uncluttered table": (
        "pick up the orange cube and place it on the plate with the right arm"
    ),
    "hit the yellow cube with the mallet using the left arm": (
        "hit the yellow cube using the mallet with the left arm"
    ),
    "hit the yellow cube with the mallet using the right arm": (
        "hit the yellow cube using the mallet with the right arm"
    ),
    "clean the plate": ("Grab the plate and the rag, and use the rag to wipe the plate."),
    "sort the items into their containers": (
        "Place the markers in the cup, the cubes in the box, and the food items on the plate."
    ),
}


@dataclasses.dataclass(frozen=True)
class LeRobotYorDataConfig(DataConfigFactory):
    """icl-dataset (bimanual YOR pick-and-place data), matching
    nyu-finger-robot/configs/yor-icl-pi05-easy-pnp-v2.yaml's data pipeline."""

    # Local dataset root (bypasses HF hub resolution -- this dataset only exists on disk).
    root: str = VICTR_DATA_ROOT
    # Episode indices to train on, precomputed via viktr.data.icl_dataset.episodes_for_task
    # for the yaml's 4 filter_tasks (filter_excluded + filter_tasks intersection). See that
    # file for how to regenerate this list if the dataset or exclusions change.
    episodes_path: str = "assets/yor_icl_pi05_easy_pnp_v2_episodes.json"
    # Passed through to DataConfig.task_prompt_overrides in create() below -- override
    # per-instance (e.g. YOR_EXPANDED_TASK_PROMPT_OVERRIDES) for a different episode set.
    task_prompt_overrides: dict[str, str] = dataclasses.field(default_factory=lambda: dict(YOR_TASK_PROMPT_OVERRIDES))
    # Drop action dims 16:20 (base_vel + lift_cmd) entirely instead of zeroing them --
    # see yor_policy.YorInputs' docstring. pi05_extended_quantilesfixed only; every
    # other config leaves this False (unchanged behavior).
    drop_base_lift: bool = False
    # Fraction of training examples for which the policy gets no proprioception --
    # forwarded to ModelTransformFactory/_transforms.TokenizePrompt (pi05's discrete
    # state-in-prompt path only; a no-op for any non-pi05 model_config). 0.0 default
    # leaves every existing config unchanged. See pi05_extended_quantilesfixed_state_
    # dropout's TrainConfig for the one config that sets this.
    state_dropout_prob: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        inputs = [yor_policy.YorInputs(drop_base_lift=self.drop_base_lift)]
        data_transforms = _transforms.Group(
            inputs=inputs,
            outputs=[yor_policy.YorOutputs(drop_base_lift=self.drop_base_lift)],
        )
        model_transforms = ModelTransformFactory(state_dropout_prob=self.state_dropout_prob)(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # icl-dataset's raw action column is "action" (singular), not the "actions"
            # default -- see yor-icl-pi05-easy-pnp-v2.yaml / tools/build_action_column.py.
            action_sequence_keys=("action",),
            # QUANTILES normalization for all Yor arms (matches create_base_config's PI05
            # default): every icl-dataset norm_stats.json (computed via openpi's own
            # scripts/compute_norm_stats.py) already carries q01/q99, and quantile norm
            # is generally more robust to outliers than MEAN_STD.
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYorDeltaRot6DDataConfig(LeRobotYorDataConfig):
    """LeRobotYorDataConfig, but with yor_policy.YorInputsDeltaRot6D/YorOutputsDeltaRot6D
    instead of YorInputs/YorOutputs -- per arm, delta cartesian position (relative to
    the query window's own first frame) + absolute orientation as the first two rows
    of the rotation matrix (Zhou et al. 2019 6D rep) instead of absolute xyz+quaternion,
    and base_vel/lift dropped entirely (same as LeRobotYorDataConfig.drop_base_lift).
    See yor_policy.py's class docstrings for the full rationale.

    This is a genuinely different action distribution from the plain arms' xyz+quat
    (delta positions and 6D rotation values don't share norm_stats with absolute
    xyz/quaternion) -- needs its own norm_stats computed fresh via
    scripts/compute_norm_stats.py, not reused via AssetsConfig."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[yor_policy.YorInputsDeltaRot6D()],
            outputs=[yor_policy.YorOutputsDeltaRot6D()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYorCanonicalDataConfig(LeRobotYorDeltaRot6DDataConfig):
    """ICRA canonical action/observation space (notes/ICRA_plan.md), with
    yor_policy.YorInputsCanonical/YorOutputsCanonical instead of
    YorInputsAlignedDeltaRot6D/YorOutputsAlignedDeltaRot6D: state is
    Hannibal52Barca/icl-dataset-fixed-obs's observation.left_ee/right_ee converted to
    absolute EE position + absolute rot6d orientation (18-dim; no gripper, no lift, no
    base) instead of raw quat+xyz (14-dim) -- matches the action side's rot6d
    orientation representation. Action encoding itself is unchanged from the parent
    class (delta EE position + absolute rot6d orientation + gripper, no lift/base).
    `root` must point at a local copy of icl-dataset-fixed-obs (same as
    LeRobotYorAlignedDeltaRot6DDataConfig) -- observation.left_ee/right_ee don't exist
    in the plain icl-dataset root.

    Genuinely different proprioception distribution from every other arm (rot6d, not
    quat) -- needs its own norm_stats computed fresh via scripts/compute_norm_stats.py,
    not reused via AssetsConfig."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[yor_policy.YorInputsCanonical()],
            outputs=[yor_policy.YorOutputsCanonical()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYorVictrDataConfig(LeRobotYorDataConfig):
    """LeRobotYorDataConfig + VICTR retrieval-context conditioning (paper Sec III-E),
    for openpi.models.pi0_victr.Pi0Victr. Query state/action/episode data is identical
    to the plain yor_icl_pi05_easy_pnp_v2 config (same episodes, same norm stats) --
    only the extra context_* fields differ, added by
    openpi.policies.yor_retrieval.RetrievalContextInputs, which must run BEFORE
    yor_policy.YorInputs (it needs raw episode_index/frame_index/prompt/camera keys
    that YorInputs replaces)."""

    # Dir written by scripts/convert_pool_for_openpi.py (viktr's outputs/victr/icl_pool,
    # re-pickled as plain dicts so it can be unpickled without importing `viktr` --
    # openpi is a separate uv workspace/venv, see yor_retrieval.py's module docstring).
    pool_dir: str = "assets/victr_icl_pool"
    retrieval_metric: yor_retrieval.RetrievalMetric = "vision"
    # Dir written by scripts/precompute_retrieval_context.py -- when set, training
    # reads precomputed context via an O(1) per-episode lookup instead of computing
    # retrieval live (DINOv2 embed + pool search + tokenization) on every sample
    # every step. None (default) preserves today's live-computation behavior exactly.
    precomputed_context_dir: str | None = None

    # RICL-style action interpolation (notes/action_interpolation.md) -- discrete
    # FAST-token blending, only valid for a pi0_fast_victr.Pi0FastVictrConfig model
    # with its own use_action_interpolation set to match (see create()'s assert
    # below). For pi0_victr.Pi0VictrConfig (continuous flow-matching, no discrete
    # action-token stream to blend), use use_continuous_action_interpolation below
    # instead -- this field must stay False there.
    use_action_interpolation: bool = False
    lamda: float = 10.0
    max_action_tokens: int = 192
    # Continuous analogue of use_action_interpolation above, for
    # pi0_victr.Pi0VictrConfig's flow-matching action head
    # (Pi0VictrConfig.use_action_interpolation; notes/action_interpolation.md) --
    # only valid for that model, with its own use_action_interpolation set to match
    # (see create()'s assert below). Reuses `lamda` above (same weighting formula,
    # same role for both mechanisms). Must stay False for Pi0FastVictrConfig, which
    # uses use_action_interpolation (discrete) instead.
    use_continuous_action_interpolation: bool = False

    # Redesigned context encoding (openpi.policies.yor_retrieval.build_context_block/
    # build_chunk_context_tokens): when set, context_frames_per_chunk camera views
    # (in this order) are read per retrieved chunk instead of context_frames_per_chunk
    # TIME frames from one camera, and context_tokens is built via the real pi0.5
    # combined task+digitized-state+truncated-action tokenization instead of the
    # legacy digit-dump text. None (default) preserves the exact legacy behavior for
    # every arm that hasn't opted in. Requires a pool built by
    # scripts/build_pool_canonical_3cam.py and context_fast_tokenizer_path set.
    context_camera_keys: tuple[str, ...] | None = None
    # Separate from Pi0FastVictrConfig.fast_model_tokenizer_kwargs (RICL's own
    # discrete-action-stream tokenizer) -- Pi0VictrConfig (continuous flow-matching)
    # has no such field, so the redesigned context encoding's tokenizer path must be
    # supplied directly here. Required when context_camera_keys is set.
    context_fast_tokenizer_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Two model configs implement the context_* fields this DataConfig wires up:
        # pi0_victr.Pi0VictrConfig (continuous flow-matching backend) and
        # pi0_fast_victr.Pi0FastVictrConfig (discrete FAST-tokenized backend, our
        # native RICL-equivalent baseline) -- both are valid here.
        assert isinstance(model_config, (pi0_victr.Pi0VictrConfig, pi0_fast_victr.Pi0FastVictrConfig))
        # Both Pi0FastVictrConfig and Pi0VictrConfig have their own
        # use_action_interpolation field (discrete FAST-token blending vs. continuous
        # flow-matching blending respectively -- see each model's docstring), so a
        # single generic getattr(..., "use_action_interpolation") can no longer
        # disambiguate which mechanism a bare match refers to; branch on the model
        # type explicitly instead.
        if isinstance(model_config, pi0_fast_victr.Pi0FastVictrConfig):
            assert model_config.use_action_interpolation == self.use_action_interpolation, (
                "LeRobotYorVictrDataConfig.use_action_interpolation must match "
                "Pi0FastVictrConfig.use_action_interpolation"
            )
            assert not self.use_continuous_action_interpolation, (
                "use_continuous_action_interpolation is Pi0VictrConfig-only; "
                "Pi0FastVictrConfig uses use_action_interpolation instead"
            )
        else:
            assert isinstance(model_config, pi0_victr.Pi0VictrConfig)
            assert model_config.use_action_interpolation == self.use_continuous_action_interpolation, (
                "LeRobotYorVictrDataConfig.use_continuous_action_interpolation must match "
                "Pi0VictrConfig.use_action_interpolation"
            )
            assert not self.use_action_interpolation, (
                "use_action_interpolation (discrete FAST-token blending) is Pi0FastVictrConfig-only; "
                "Pi0VictrConfig has no discrete action-token stream to blend -- use "
                "use_continuous_action_interpolation instead"
            )
        # Loaded once here (not deferred) so RetrievalContextInputs can quantile-
        # normalize the neighbor's actions to the same scale the query's own actions
        # go through via the Normalize transform added downstream in data_loader.py --
        # see RetrievalContextInputs.action_norm_stats' docstring. Reused below instead
        # of calling create_base_config a second time.
        base = self.create_base_config(assets_dirs, model_config)
        data_transforms = _transforms.Group(
            inputs=[
                yor_retrieval.RetrievalContextInputs(
                    pool_dir=self.pool_dir,
                    icl_dataset_root=self.root,
                    primary_camera="observation.images.zed",  # matches viktr.data.icl_dataset.PRIMARY_CAMERA
                    retrieval_metric=self.retrieval_metric,
                    num_context_chunks=model_config.num_context_chunks,
                    context_frames_per_chunk=model_config.context_frames_per_chunk,
                    context_text_max_length=model_config.context_text_max_length,
                    precomputed_dir=self.precomputed_context_dir,
                    use_action_interpolation=self.use_action_interpolation,
                    use_continuous_action_interpolation=self.use_continuous_action_interpolation,
                    lamda=self.lamda,
                    max_action_tokens=self.max_action_tokens,
                    fast_tokenizer_path=(model_config.fast_model_tokenizer_kwargs or {}).get("fast_tokenizer_path")
                    if self.use_action_interpolation
                    else self.context_fast_tokenizer_path,
                    action_horizon=model_config.action_horizon if self.use_action_interpolation else None,
                    drop_base_lift=self.drop_base_lift if self.use_action_interpolation else False,
                    action_norm_stats=(base.norm_stats or {}).get("actions")
                    if (
                        self.use_action_interpolation
                        or self.use_continuous_action_interpolation
                        or self.context_camera_keys is not None
                    )
                    else None,
                    # See RetrievalContextInputs.prompt_to_task_overrides' docstring --
                    # only matters for the live-retrieval path (precomputed_context_dir=
                    # None); harmless (empty dict -> None) for the precomputed path.
                    prompt_to_task_overrides={v: k for k, v in self.task_prompt_overrides.items()} or None,
                    context_camera_keys=self.context_camera_keys,
                    state_norm_stats=(base.norm_stats or {}).get("state") if self.context_camera_keys is not None else None,
                ),
                yor_policy.YorInputs(drop_base_lift=self.drop_base_lift),
            ],
            outputs=[yor_policy.YorOutputs(drop_base_lift=self.drop_base_lift)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            base,
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            # QUANTILES for all Yor arms -- see LeRobotYorDataConfig.create()'s comment.
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYorVictrCanonicalDataConfig(LeRobotYorVictrDataConfig):
    """LeRobotYorVictrDataConfig, but with yor_policy.YorInputsCanonical/
    YorOutputsCanonical instead of YorInputs/YorOutputs -- the 3 VICTR retrieval arms
    (vision/value/vision_value) plus RICL (Pi0FastVictrConfig + action interpolation)
    all read the same ICRA canonical action/observation space as
    LeRobotYorCanonicalDataConfig (notes/ICRA_plan.md): state is absolute EE
    position + absolute rot6d orientation (18-dim, no gripper/lift/base), action is
    delta EE position + absolute rot6d orientation + gripper (20-dim, no lift/base).
    RetrievalContextInputs wiring (pool_dir/retrieval_metric/precomputed_context_dir/
    use_action_interpolation/use_continuous_action_interpolation/lamda/
    max_action_tokens) is unchanged from the parent class -- it's agnostic to the
    query's own action/state encoding. `root` must point at a local copy of
    icl-dataset-fixed-obs, same as LeRobotYorCanonicalDataConfig."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        assert isinstance(model_config, (pi0_victr.Pi0VictrConfig, pi0_fast_victr.Pi0FastVictrConfig))
        # See the parent class's create() for why this can no longer be a single
        # generic getattr check -- both model configs now have their own
        # use_action_interpolation field (discrete vs. continuous).
        if isinstance(model_config, pi0_fast_victr.Pi0FastVictrConfig):
            assert model_config.use_action_interpolation == self.use_action_interpolation, (
                "LeRobotYorVictrCanonicalDataConfig.use_action_interpolation must match "
                "Pi0FastVictrConfig.use_action_interpolation"
            )
            assert not self.use_continuous_action_interpolation, (
                "use_continuous_action_interpolation is Pi0VictrConfig-only; "
                "Pi0FastVictrConfig uses use_action_interpolation instead"
            )
        else:
            assert isinstance(model_config, pi0_victr.Pi0VictrConfig)
            assert model_config.use_action_interpolation == self.use_continuous_action_interpolation, (
                "LeRobotYorVictrCanonicalDataConfig.use_continuous_action_interpolation must match "
                "Pi0VictrConfig.use_action_interpolation"
            )
            assert not self.use_action_interpolation, (
                "use_action_interpolation (discrete FAST-token blending) is Pi0FastVictrConfig-only; "
                "Pi0VictrConfig has no discrete action-token stream to blend -- use "
                "use_continuous_action_interpolation instead"
            )
        base = self.create_base_config(assets_dirs, model_config)
        data_transforms = _transforms.Group(
            inputs=[
                yor_retrieval.RetrievalContextInputs(
                    pool_dir=self.pool_dir,
                    icl_dataset_root=self.root,
                    primary_camera="observation.images.zed",
                    retrieval_metric=self.retrieval_metric,
                    num_context_chunks=model_config.num_context_chunks,
                    context_frames_per_chunk=model_config.context_frames_per_chunk,
                    context_text_max_length=model_config.context_text_max_length,
                    precomputed_dir=self.precomputed_context_dir,
                    use_action_interpolation=self.use_action_interpolation,
                    use_continuous_action_interpolation=self.use_continuous_action_interpolation,
                    lamda=self.lamda,
                    max_action_tokens=self.max_action_tokens,
                    fast_tokenizer_path=(model_config.fast_model_tokenizer_kwargs or {}).get("fast_tokenizer_path")
                    if self.use_action_interpolation
                    else self.context_fast_tokenizer_path,
                    action_horizon=model_config.action_horizon if self.use_action_interpolation else None,
                    drop_base_lift=self.drop_base_lift if self.use_action_interpolation else False,
                    # The pool's `actions` are already ICRA-canonical (delta-pos + rot6d
                    # + gripper, no base/lift dims) -- see
                    # yor_retrieval.RetrievalContextInputs.canonical_actions' docstring.
                    # Only meaningful for the discrete path (continuous_action_
                    # interpolation_extras always assumes canonical, see its docstring).
                    canonical_actions=self.use_action_interpolation,
                    action_norm_stats=(base.norm_stats or {}).get("actions")
                    if (
                        self.use_action_interpolation
                        or self.use_continuous_action_interpolation
                        or self.context_camera_keys is not None
                    )
                    else None,
                    prompt_to_task_overrides={v: k for k, v in self.task_prompt_overrides.items()} or None,
                    context_camera_keys=self.context_camera_keys,
                    state_norm_stats=(base.norm_stats or {}).get("state") if self.context_camera_keys is not None else None,
                ),
                yor_policy.YorInputsCanonical(),
            ],
            outputs=[yor_policy.YorOutputsCanonical()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            base,
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        # Stage 1: pi0.5 fine-tuned from pi05_base on the 20-task / 1,186-episode canonical set
        # (pos + rot6d state; delta-pos + rot6d + gripper actions). Its step-40k checkpoint
        # initializes the ctx3cam VICTR arms and the warm-start baseline.
        name="yor_icl_pi05_canonical_extended",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # VICTR, vision retrieval: top-1 chunk by DINOv2 distance; 3 camera views per context chunk +
        # FAST-tokenized state/action text. Needs shells/precompute_context.sh ctx3cam vision.
        name="yor_icl_victr_vision_canonical_ctx3cam",
        model=pi0_victr.Pi0VictrConfig(
            pi05=True,
            action_horizon=30,
            retrieval_metric="vision",
            num_context_chunks=1,
            context_frames_per_chunk=3,
            context_text_max_length=256,
        ),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_3cam_224",
            precomputed_context_dir=f"{VICTR_RETRIEVAL_CONTEXT_DIR}/canonical_3cam_k1",
            context_camera_keys=(
                "observation.images.zed",
                "observation.images.fish0",
                "observation.images.fish1",
            ),
            context_fast_tokenizer_path=VICTR_FAST_TOKENIZER,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=30_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            warmup_init_lr=9.65e-6,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader(VICTR_WARMSTART_PARAMS),
    ),
    TrainConfig(
        # VICTR, value retrieval: top-1 chunk by progress-value alignment.
        # Needs shells/precompute_context.sh ctx3cam value.
        name="yor_icl_victr_value_canonical_ctx3cam",
        model=pi0_victr.Pi0VictrConfig(
            pi05=True,
            action_horizon=30,
            retrieval_metric="value",
            num_context_chunks=1,
            context_frames_per_chunk=3,
            context_text_max_length=256,
        ),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="value",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_3cam_224",
            precomputed_context_dir=f"{VICTR_RETRIEVAL_CONTEXT_DIR}/canonical_3cam_k1",
            context_camera_keys=(
                "observation.images.zed",
                "observation.images.fish0",
                "observation.images.fish1",
            ),
            context_fast_tokenizer_path=VICTR_FAST_TOKENIZER,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=30_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            warmup_init_lr=9.65e-6,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader(VICTR_WARMSTART_PARAMS),
    ),
    TrainConfig(
        # VICTR, fused vision + value retrieval.
        # Needs shells/precompute_context.sh ctx3cam vision_value.
        name="yor_icl_victr_vision_value_canonical_ctx3cam",
        model=pi0_victr.Pi0VictrConfig(
            pi05=True,
            action_horizon=30,
            retrieval_metric="vision_value",
            num_context_chunks=1,
            context_frames_per_chunk=3,
            context_text_max_length=256,
        ),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision_value",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_3cam_224",
            precomputed_context_dir=f"{VICTR_RETRIEVAL_CONTEXT_DIR}/canonical_3cam_k1",
            context_camera_keys=(
                "observation.images.zed",
                "observation.images.fish0",
                "observation.images.fish1",
            ),
            context_fast_tokenizer_path=VICTR_FAST_TOKENIZER,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=30_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            warmup_init_lr=9.65e-6,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader(VICTR_WARMSTART_PARAMS),
    ),
    TrainConfig(
        # RICL-style baseline on the VICTR architecture: vision retrieval plus continuous action
        # interpolation toward the retrieved chunk's actions (weight exp(-lamda * distance)).
        # Needs shells/precompute_context.sh ctx3cam_interp.
        name="yor_icl_victr_vision_interp_canonical_ctx3cam",
        model=pi0_victr.Pi0VictrConfig(
            pi05=True,
            action_horizon=30,
            retrieval_metric="vision",
            num_context_chunks=1,
            context_frames_per_chunk=3,
            context_text_max_length=256,
            use_action_interpolation=True,
            lamda=3.0,
        ),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_3cam_224",
            precomputed_context_dir=f"{VICTR_RETRIEVAL_CONTEXT_DIR}/canonical_3cam_k1_interp",
            use_continuous_action_interpolation=True,
            lamda=3.0,
            context_camera_keys=(
                "observation.images.zed",
                "observation.images.fish0",
                "observation.images.fish1",
            ),
            context_fast_tokenizer_path=VICTR_FAST_TOKENIZER,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=30_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            warmup_init_lr=9.65e-6,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader(VICTR_WARMSTART_PARAMS),
    ),
    TrainConfig(
        # Plain pi0.5 baseline with the same warm start and schedule as the ctx3cam VICTR arms
        # (no retrieved context).
        name="yor_icl_pi05_canonical_warmstart_baseline",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=30_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            warmup_init_lr=9.65e-6,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(VICTR_WARMSTART_PARAMS),
    ),
    TrainConfig(
        # RICL baseline: pi0-FAST with vision-retrieved context and discrete action interpolation
        # (arXiv:2508.02062). Needs shells/precompute_context.sh fast_interp.
        name="yor_icl_fast_victr_vision_interp_canonical",
        model=pi0_fast_victr.Pi0FastVictrConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            num_context_chunks=1,
            context_frames_per_chunk=1,
            context_text_max_length=1024,
            retrieval_metric="vision",
            use_action_interpolation=True,
            lamda=10.0,
            max_action_tokens=224,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": VICTR_FAST_TOKENIZER,
            },
        ),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_224",
            precomputed_context_dir=f"{VICTR_RETRIEVAL_CONTEXT_DIR}/canonical_interp",
            use_action_interpolation=True,
            lamda=10.0,
            max_action_tokens=224,
        ),
        batch_size=64,
        num_workers=8,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    ),
    TrainConfig(
        # Plain pi0-FAST baseline on the canonical set, with the fitted canonical FAST tokenizer.
        name="yor_icl_pi0_fast_canonical",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": VICTR_FAST_TOKENIZER,
            },
        ),
        data=LeRobotYorCanonicalDataConfig(
            repo_id="icl-dataset",
            root=VICTR_DATA_ROOT,
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
