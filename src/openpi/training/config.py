"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_ki as pi0_ki
import openpi.models.pi0_victr as pi0_victr
import openpi.models.pi0_fast_victr as pi0_fast_victr
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.yor_policy as yor_policy
import openpi.policies.yor_ki as yor_ki
import openpi.policies.yor_rabc as yor_rabc
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
    root: str = "/scratch/lim2045/icl_ws/icl-dataset"
    # Episode indices to train on, precomputed via viktr.data.icl_dataset.episodes_for_task
    # for the yaml's 4 filter_tasks (filter_excluded + filter_tasks intersection). See that
    # file for how to regenerate this list if the dataset or exclusions change.
    episodes_path: str = "assets/yor_icl_pi05_easy_pnp_v2_episodes.json"
    # Passed through to DataConfig.task_prompt_overrides in create() below -- override
    # per-instance (e.g. YOR_EXPANDED_TASK_PROMPT_OVERRIDES) for a different episode set.
    task_prompt_overrides: dict[str, str] = dataclasses.field(default_factory=lambda: dict(YOR_TASK_PROMPT_OVERRIDES))
    # Dir written by scripts/precompute_rabc_weights.py (viktr-side) -- when set,
    # yor_rabc.RabcWeightInputs is prepended to data_transforms.inputs, adding a
    # per-item RA-BC loss weight (notes/reward_aligned_bc.md) that scripts/train.py's
    # loss_fn reads. None (default) leaves Observation.rabc_weight unset, i.e. plain
    # unweighted BC -- exactly today's behavior.
    rabc_weights_dir: str | None = None
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
        if self.rabc_weights_dir is not None:
            # Must run before YorInputs -- needs the raw episode_index/frame_index keys
            # YorInputs replaces (see RabcWeightInputs' docstring).
            inputs.insert(0, yor_rabc.RabcWeightInputs(weights_dir=self.rabc_weights_dir))
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
class LeRobotYorAbsoluteJointDataConfig(LeRobotYorDataConfig):
    """LeRobotYorDataConfig, but with yor_policy.YorInputsAbsoluteJoint/
    YorOutputsAbsoluteJoint instead of YorInputs/YorOutputs -- action becomes the
    future window of observation.state itself (icl-dataset's 15-dim absolute joint
    positions: left/right arm joints + lift), matching the observation space, instead
    of YorInputs' 20-dim EE-pose action. See yor_policy.YorInputsAbsoluteJoint's
    docstring for why action_sequence_keys points at observation.state rather than
    action.

    Genuinely different action distribution from the plain arms' EE xyz+quat (joint
    positions don't share norm_stats with EE poses) -- needs its own norm_stats
    computed fresh via scripts/compute_norm_stats.py, not reused via AssetsConfig
    (same convention as LeRobotYorDeltaRot6DDataConfig)."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[yor_policy.YorInputsAbsoluteJoint()],
            outputs=[yor_policy.YorOutputsAbsoluteJoint()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # The action_sequence_keys difference from LeRobotYorDataConfig's ("action",)
            # -- see yor_policy.YorInputsAbsoluteJoint's docstring for why this works.
            action_sequence_keys=("observation.state",),
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYorAlignedQDataConfig(LeRobotYorDataConfig):
    """LeRobotYorDataConfig, but with yor_policy.YorInputsAlignedQ/YorOutputsAlignedQ
    instead of YorInputs/YorOutputs, reading a genuinely aligned absolute-joint
    action/proprioception pair from Hannibal52Barca/icl-dataset-fixed-action:
    state is observation.state with the lift dim dropped (14-dim: left/right
    arm joints only -- no base component exists in observation.state to begin
    with), action is that dataset's new action.q_target (14-dim,
    IK-reconstructed from action.left_ee/right_ee and anchored per-episode to
    observation.state's own FK -- see that dataset's README) with gripper
    appended from the raw "action" column's grip dims (q_target itself has no
    gripper channel), giving a 16-dim action with no lift_cmd/base_vel either
    -- unlike LeRobotYorAbsoluteJointDataConfig, this is a REAL recorded
    action column, not observation.state's own future window reused as a
    pseudo-action. `root` must point at a local copy of icl-dataset-fixed-action
    (built via nyu-finger-robot/tools/merge_hf_aligned_columns.py), not the plain
    icl-dataset root -- action.q_target doesn't exist there.

    Genuinely different action distribution from every other arm -- needs its
    own norm_stats computed fresh via scripts/compute_norm_stats.py, not reused
    via AssetsConfig (same convention as LeRobotYorAbsoluteJointDataConfig /
    LeRobotYorDeltaRot6DDataConfig)."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[yor_policy.YorInputsAlignedQ()],
            outputs=[yor_policy.YorOutputsAlignedQ()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        episodes = json.loads(pathlib.Path(self.episodes_path).read_text())
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            root=self.root,
            episodes=episodes,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # ("action",) alongside q_target -- YorInputsAlignedQ appends gripper
            # from the raw action column's grip dims (q_target has none).
            action_sequence_keys=("action.q_target", "action"),
            use_quantile_norm=True,
            task_prompt_overrides=self.task_prompt_overrides,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYorAlignedDeltaRot6DDataConfig(LeRobotYorDeltaRot6DDataConfig):
    """LeRobotYorDeltaRot6DDataConfig, but with
    yor_policy.YorInputsAlignedDeltaRot6D/YorOutputsAlignedDeltaRot6D instead of
    YorInputsDeltaRot6D/YorOutputsDeltaRot6D: state comes from
    Hannibal52Barca/icl-dataset-fixed-obs's observation.left_ee/right_ee
    (absolute EE pose, FK'd from the real joint encoders) instead of
    observation.state (joint space) -- aligns proprioception with the
    EE-space delta-position + absolute-rot6d-orientation action, instead of
    mixing joint-space state with EE-space action. Action encoding itself is
    unchanged from the parent class. `root` must point at a local copy of
    icl-dataset-fixed-obs (built via nyu-finger-robot/tools/
    merge_hf_aligned_columns.py), not the plain icl-dataset root --
    observation.left_ee/right_ee don't exist there.

    Genuinely different proprioception distribution from every other arm --
    needs its own norm_stats computed fresh via scripts/compute_norm_stats.py,
    not reused via AssetsConfig."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[yor_policy.YorInputsAlignedDeltaRot6D()],
            outputs=[yor_policy.YorOutputsAlignedDeltaRot6D()],
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
class LeRobotYorKiDataConfig(LeRobotYorDataConfig):
    """LeRobotYorDataConfig + Knowledge Insulation / subtask-prediction auxiliary
    targets, for openpi.models.pi0_ki.Pi0Ki. Query state/action/episode data is
    identical to the plain yor_icl_pi05_easy_pnp_v2 config (same episodes) -- only the
    extra fast_action_tokens/subtask_tokens fields differ, added by
    openpi.policies.yor_ki.KiTargetInputs, which must run BEFORE yor_policy.YorInputs
    (it needs raw episode_index/frame_index/action keys that YorInputs replaces).

    Uses QUANTILES normalization (unlike the plain/VICTR arms' MEAN_STD), matching
    lerobot's own KI ablation and required for the FAST tokenizer -- fit on
    QUANTILES-normalized action chunks -- to see the numeric range it was calibrated
    for. See yor_ki.py's module docstring.

    Also composes with rabc_weights_dir (inherited from LeRobotYorDataConfig) -- see
    create()'s comment for how RA-BC's per-item loss weight and KI's aux losses split."""

    fast_tokenizer_path: str = (
        "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
        "yor-icl-pi05-easy-pnp-v2"
    )
    max_action_tokens: int = 256
    use_subtask_prediction: bool = True
    subtask_annotations_path: str = "/scratch/lim2045/icl_ws/icl-dataset/meta/subtask_annotations/subtasks.jsonl"
    max_subtask_tokens: int = 32

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        assert isinstance(model_config, pi0_ki.Pi0KiConfig)
        # Called first (unlike LeRobotYorVictrDataConfig) so KiTargetInputs can read the
        # actions norm stats it needs to quantile-normalize action chunks before
        # FAST-tokenizing them.
        base = self.create_base_config(assets_dirs, model_config)
        action_stats = base.norm_stats["actions"]
        inputs = [
            yor_ki.KiTargetInputs(
                fast_tokenizer_path=self.fast_tokenizer_path,
                max_action_tokens=self.max_action_tokens,
                action_q01=tuple(float(v) for v in action_stats.q01),
                action_q99=tuple(float(v) for v in action_stats.q99),
                use_subtask_prediction=self.use_subtask_prediction,
                subtask_annotations_path=self.subtask_annotations_path,
                max_subtask_tokens=self.max_subtask_tokens,
            ),
            yor_policy.YorInputs(),
        ]
        if self.rabc_weights_dir is not None:
            # RA-BC + KI compose without any model-side change: scripts/train.py's
            # loss_fn RABC-weights whatever compute_loss returns per item, and
            # Pi0Ki.compute_loss returns flow_loss (per-item, shape [b, ah]) +
            # aux_loss (ki_loss/subtask_loss, a single batch-uniform scalar computed
            # by _next_token_ce's own sum/mask reduction -- the same value added to
            # every item). Since a constant's weighted mean equals the constant, the
            # RABC weighting ends up applying only to the flow-matching term -- the
            # discrete KI/subtask CE losses stay plain batch-uniform, unaffected by
            # per-item reward alignment. That is the intended split: RA-BC targets
            # "how much to trust this demonstration's actions", not the auxiliary
            # representation-learning objectives. Must run before YorInputs, same
            # requirement as LeRobotYorDataConfig.create() (needs raw episode_index/
            # frame_index, which YorInputs replaces).
            inputs.insert(0, yor_rabc.RabcWeightInputs(weights_dir=self.rabc_weights_dir))
        data_transforms = _transforms.Group(
            inputs=inputs,
            outputs=[yor_policy.YorOutputs()],
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

    # RICL-style action interpolation (notes/action_interpolation.md) -- only valid
    # for a pi0_fast_victr.Pi0FastVictrConfig model with its own use_action_interpolation
    # set to match (see create()'s assert below); ignored (must stay False) for
    # pi0_victr.Pi0VictrConfig, which has no discrete action-token stream to blend.
    use_action_interpolation: bool = False
    lamda: float = 10.0
    max_action_tokens: int = 192

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Two model configs implement the context_* fields this DataConfig wires up:
        # pi0_victr.Pi0VictrConfig (continuous flow-matching backend) and
        # pi0_fast_victr.Pi0FastVictrConfig (discrete FAST-tokenized backend, our
        # native RICL-equivalent baseline) -- both are valid here.
        assert isinstance(model_config, (pi0_victr.Pi0VictrConfig, pi0_fast_victr.Pi0FastVictrConfig))
        # rabc_weights_dir is inherited from LeRobotYorDataConfig but not wired up
        # below -- fail loudly instead of silently ignoring it if ever set here.
        assert self.rabc_weights_dir is None, "RA-BC + VICTR retrieval is not implemented; rabc_weights_dir must be None"
        use_action_interpolation = getattr(model_config, "use_action_interpolation", False)
        assert use_action_interpolation == self.use_action_interpolation, (
            "LeRobotYorVictrDataConfig.use_action_interpolation must match "
            "model_config.use_action_interpolation (only Pi0FastVictrConfig has the latter; "
            "Pi0VictrConfig has no discrete action-token stream to blend, so both must be False there)"
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
                    lamda=self.lamda,
                    max_action_tokens=self.max_action_tokens,
                    fast_tokenizer_path=(model_config.fast_model_tokenizer_kwargs or {}).get("fast_tokenizer_path")
                    if self.use_action_interpolation
                    else None,
                    action_horizon=model_config.action_horizon if self.use_action_interpolation else None,
                    drop_base_lift=self.drop_base_lift if self.use_action_interpolation else False,
                    action_norm_stats=(base.norm_stats or {}).get("actions") if self.use_action_interpolation else None,
                    # See RetrievalContextInputs.prompt_to_task_overrides' docstring --
                    # only matters for the live-retrieval path (precomputed_context_dir=
                    # None); harmless (empty dict -> None) for the precomputed path.
                    prompt_to_task_overrides={v: k for k, v in self.task_prompt_overrides.items()} or None,
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
    use_action_interpolation/lamda/max_action_tokens) is unchanged from the parent
    class -- it's agnostic to the query's own action/state encoding. `root` must point
    at a local copy of icl-dataset-fixed-obs, same as LeRobotYorCanonicalDataConfig."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        assert isinstance(model_config, (pi0_victr.Pi0VictrConfig, pi0_fast_victr.Pi0FastVictrConfig))
        assert self.rabc_weights_dir is None, "RA-BC + VICTR retrieval is not implemented; rabc_weights_dir must be None"
        use_action_interpolation = getattr(model_config, "use_action_interpolation", False)
        assert use_action_interpolation == self.use_action_interpolation, (
            "LeRobotYorVictrCanonicalDataConfig.use_action_interpolation must match "
            "model_config.use_action_interpolation (only Pi0FastVictrConfig has the latter; "
            "Pi0VictrConfig has no discrete action-token stream to blend, so both must be False there)"
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
                    lamda=self.lamda,
                    max_action_tokens=self.max_action_tokens,
                    fast_tokenizer_path=(model_config.fast_model_tokenizer_kwargs or {}).get("fast_tokenizer_path")
                    if self.use_action_interpolation
                    else None,
                    action_horizon=model_config.action_horizon if self.use_action_interpolation else None,
                    drop_base_lift=self.drop_base_lift if self.use_action_interpolation else False,
                    # The pool's `actions` are already ICRA-canonical (delta-pos + rot6d
                    # + gripper, no base/lift dims) -- see
                    # yor_retrieval.RetrievalContextInputs.canonical_actions' docstring.
                    canonical_actions=self.use_action_interpolation,
                    action_norm_stats=(base.norm_stats or {}).get("actions") if self.use_action_interpolation else None,
                    prompt_to_task_overrides={v: k for k, v in self.task_prompt_overrides.items()} or None,
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
        # Reproduces nyu-finger-robot/configs/yor-icl-pi05-easy-pnp-v2.yaml (the lerobot-
        # based pi05 fine-tune on icl-dataset's 4-task pick-and-place subset) on openpi's
        # own JAX pi05 implementation, to cross-check lerobot's pi05 training pipeline.
        name="yor_icl_pi05_easy_pnp_v2",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),  # yaml's chunk_size/n_action_steps
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
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
        # lerobot's pi05 implementation has no EMA; matching that here for a fair
        # comparison (openpi's own default is ema_decay=0.99).
        ema_decay=None,
        save_interval=5_000,  # yaml's save_freq (its eval_freq is a lerobot-only no-op)
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Re-run of the earlier "sanity15k" check (checkpoints/yor_icl_pi05_easy_pnp_v2/
        # yor_icl_pi05_easy_pnp_v2_sanity15k/) with a correctly-scaled LR schedule. That
        # run reused yor_icl_pi05_easy_pnp_v2's TrainConfig with num_train_steps
        # CLI-overridden to 15_000, but its lr_schedule.decay_steps was left at 50_000 --
        # so the cosine decay only got 15/50 = 30% of the way through its span (~27%
        # of the way through the post-warmup decay portion), landing at ~4.23e-5 by the
        # final step -- 85% of peak_lr (5e-5), nowhere near the 5e-6 decay_lr floor a
        # schedule actually built for a 15k-step run would reach. Everything else
        # (data/episodes, batch size, optimizer, weight_loader) is identical to
        # yor_icl_pi05_easy_pnp_v2 -- only num_train_steps and decay_steps are fixed to
        # match. Same episodes_path as yor_icl_pi05_easy_pnp_v2, so
        # assets/sanity15k_annealing/icl-dataset/norm_stats.json is a copy of that
        # config's norm_stats.json (same convention as the other single-run arms in
        # this file).
        name="sanity15k_annealing",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Continuation of the earlier "sanity15k" (no-annealing) run -- checkpoints/
        # yor_icl_pi05_easy_pnp_v2/yor_icl_pi05_easy_pnp_v2_sanity15k/ -- not
        # sanity15k_annealing. Eval showed the no-annealing run (LR only ~15% decayed
        # off peak by step 15k, decay_steps left at 50k -- see sanity15k_annealing's
        # comment) slightly outperformed the correctly-annealed one, on the theory
        # that the annealed run's LR decayed too far, too fast to keep fitting the
        # data in only 15k steps. Rather than guess at a slower decay from scratch,
        # this resumes from the no-annealing run's final (step 14999) checkpoint and
        # trains a further 15k steps with its own from-scratch schedule: no warmup
        # (already warmed up during the first 15k steps) and a cosine decay that
        # actually reaches its floor over this second 15k-step span (peak_lr/decay_lr
        # match every other yor_icl_pi05_* arm's schedule for comparability).
        #
        # weight_loader points at the sanity15k run's own on-disk checkpoint instead
        # of the pi05_base release checkpoint every other arm in this file starts
        # from -- CheckpointWeightLoader accepts either (see its docstring). Only the
        # params are seeded this way -- num_train_steps=15_000 is this run's OWN step
        # count (0..15000, not 15000..30000); the source run's optimizer state/step/
        # wandb id are not carried over.
        #
        # Same episodes/data as yor_icl_pi05_easy_pnp_v2 (LeRobotYorDataConfig's
        # default episodes_path), so assets/yor_icl_pi05_sanity15k_continued_15k/
        # icl-dataset/norm_stats.json is a copy of that config's norm_stats.json, not
        # computed fresh (same convention as this file's other single-run arms).
        name="yor_icl_pi05_sanity15k_continued_15k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/yor_icl_pi05_easy_pnp_v2/yor_icl_pi05_easy_pnp_v2_sanity15k/14999/params"
        ),
    ),
    TrainConfig(
        # Reward-Aligned Behavior Cloning (notes/reward_aligned_bc.md, SARM paper Sec
        # 3.2) applied to yor_icl_pi05_easy_pnp_v2's exact data/hyperparameters --
        # isolates RA-BC's effect from VICTR retrieval (see reward_aligned_bc.md Sec 7's
        # validation plan). Same episodes_path (same 196-episode 4-task pnp subset) as
        # yor_icl_pi05_easy_pnp_v2, so assets/yor_icl_pi05_rabc/icl-dataset/norm_stats.json
        # is a copy of that config's norm_stats.json, not computed fresh (identical
        # state/action distributions -- same convention as yor_icl_victr_value_expanded's
        # norm_stats copy, see that config's comment).
        #
        # Needs assets/rabc_weights/yor_icl_pi05_rabc/*.npy before its first training
        # job -- run once via scripts/precompute_rabc_weights.py (viktr-side venv):
        #   uv run python scripts/precompute_rabc_weights.py \
        #       --episodes-path third_party/openpi/assets/yor_icl_pi05_easy_pnp_v2_episodes.json \
        #       --out-dir third_party/openpi/assets/rabc_weights/yor_icl_pi05_rabc
        # (delta=30 default matches action_horizon below). Otherwise identical to
        # yor_icl_pi05_easy_pnp_v2, so the two arms are directly comparable.
        name="yor_icl_pi05_rabc",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            rabc_weights_dir="assets/rabc_weights/yor_icl_pi05_rabc",
        ),
        batch_size=128,
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
        # Training-Time RTC (notes/training_time_rtc.md; "Training-Time Action
        # Conditioning for Efficient Real-Time Chunking", arXiv 2512.05964, reference impl
        # third_party/real-time-chunking-kinetix) applied to yor_icl_pi05_easy_pnp_v2's
        # exact data/hyperparameters -- isolates training-time RTC's effect the same way
        # yor_icl_pi05_rabc isolates RA-BC's. All of the mechanism lives in
        # Pi0Config.simulated_delay / Pi0.compute_loss (per-example simulated inference
        # delay, action-chunk prefix pinned to the ground-truth action, loss masked to the
        # noisy suffix) and the generalization of gemma.RMSNorm's adaRMS conditioning to
        # per-action-chunk-position (openpi.models.gemma.RMSNorm, Pi0.embed_suffix) -- no
        # data-side changes, so this reuses LeRobotYorDataConfig directly (same episodes,
        # same norm_stats.json as yor_icl_pi05_easy_pnp_v2, copied not recomputed -- same
        # convention as yor_icl_victr_value_expanded's norm_stats copy).
        #
        # simulated_delay=19 mirrors the reference repo's ratio (simulated_delay=5 out of
        # action_chunk_size=8, ~62% of the chunk) scaled to this config's
        # action_horizon=30 (round(30 * 5 / 8) == 19); the delay actually sampled during
        # training is exponentially weighted towards small values, so the *expected*
        # resolved prefix is well under 19. Trained from pi05_base for the full 50k steps
        # (not fine-tuned from yor_icl_pi05_easy_pnp_v2's checkpoint) to stay directly,
        # apples-to-apples comparable to the other pi05 arms on this dataset.
        #
        # Inference-time real-time execution (conditioning sample_actions on a previous
        # chunk's committed prefix, i.e. actually exercising what this training teaches)
        # is not implemented yet -- follow-up work, not needed to train this arm.
        name="yor_icl_pi05_rtc",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, simulated_delay=19),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
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
        # Sanity check ahead of yor_icl_pi0_fast_expanded (below, in the expanded-data
        # section): pi0-FAST analogue of yor_icl_pi05_easy_pnp_v2, run on the SAME small
        # 4-task pick-and-place subset (not the 31-task expanded set the full run below
        # trains on) so the FAST-tokenized backend/data pipeline can be validated cheaply
        # (15k steps) before committing to a full-length run on the bigger dataset.
        # decay_steps matched to num_train_steps (same annealing fix sanity15k_annealing
        # applied to yor_icl_pi05_easy_pnp_v2 -- see that config's comment for why
        # decay_steps must match num_train_steps rather than being left at a longer run's
        # value).
        #
        # action_dim=20/action_horizon=30 match icl-dataset's actual action shape (same
        # as every other yor_icl_* arm -- see yor_policy.YorInputs); max_token_len=256
        # matches yor_icl_fast_victr_vision_expanded's choice for the identical action
        # shape (that config's comment found action-only FAST tokens can reach ~191).
        # fast_tokenizer_path points at the tokenizer nyu-finger-robot/tools/fixes/
        # fit_fast_tokenizer.py already fit on this exact 4-task subset's action
        # distribution (outputs/fast_tokenizer/yor-icl-pi05-easy-pnp-v2/, see its sibling
        # _token_budget_full.json: action-span p99/max of 104/191 tokens, prefix
        # p99/max of 82/83 -- 256 leaves headroom over both combined at the tails).
        #
        # Reuses yor_icl_pi05_easy_pnp_v2's norm_stats.json (copied to
        # assets/yor_icl_pi0_fast_easy_pnp_v2_sanity15k_annealing/icl-dataset/
        # norm_stats.json -- same episodes, so identical state/action distributions;
        # same convention as the other single-run arms in this file).
        name="yor_icl_pi0_fast_easy_pnp_v2_sanity15k_annealing",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-pi05-easy-pnp-v2"
                ),
            },
        ),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    ),
    TrainConfig(
        # Sanity check for LeRobotYorAbsoluteJointDataConfig: both state AND action are
        # icl-dataset's 15-dim observation.state (absolute joint positions), instead of
        # every other yor_icl_* arm's 20-dim EE-pose action -- see that class's and
        # yor_policy.YorInputsAbsoluteJoint's docstrings for how the action is derived
        # (icl-dataset never recorded a joint-space action; this uses the future window
        # of observation.state itself). Run on the SAME small 4-task pick-and-place
        # subset as yor_icl_pi05_easy_pnp_v2 (not the expanded set) to validate the new
        # data pipeline cheaply (15k steps) before committing to a longer run.
        #
        # decay_steps matched to num_train_steps (same annealing fix sanity15k_annealing
        # applied to yor_icl_pi05_easy_pnp_v2 -- see that config's comment for why
        # decay_steps must match num_train_steps rather than being left at a longer
        # run's value).
        #
        # Genuinely different action distribution from every EE-pose arm (joint
        # positions, not xyz+quat) -- needs its own norm_stats, computed fresh via
        # scripts/compute_norm_stats.py (run once before this config's first training
        # job; not reused via AssetsConfig -- see LeRobotYorAbsoluteJointDataConfig's
        # docstring).
        name="yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorAbsoluteJointDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Same as yor_icl_pi05_easy_pnp_v2, but trained on viktr.data.icl_dataset.
        # expanded_tasks() (31 of icl-dataset's 36 tasks, Hub keep-field-curated,
        # 1,784 episodes) instead of the original 4-task pick-and-place subset. Needs
        # its own norm_stats (assets/yor_icl_pi05_expanded/icl-dataset/norm_stats.json,
        # computed via scripts/compute_norm_stats.py -- run once before this config's
        # first training job).
        name="yor_icl_pi05_expanded",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=128,
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
        # yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k's absolute-joint-state
        # action space (see LeRobotYorAbsoluteJointDataConfig/
        # yor_policy.YorInputsAbsoluteJoint), scaled up the same way
        # yor_icl_pi05_expanded scales up yor_icl_pi05_easy_pnp_v2: full 50k steps on
        # viktr.data.icl_dataset.expanded_tasks() (31 of icl-dataset's 36 tasks,
        # 1,784 episodes) instead of the 4-task/196-episode sanity subset. The
        # sanity15k run (12.4GB checkpoint at lair-nyu/
        # yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k) came back clean (loss
        # ~7e-4 by step 15k on the small subset) -- this is the "commit to a longer
        # run" scale-up that sanity check was gating.
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_expanded_absolute_joint/
        # icl-dataset/norm_stats.json, computed via scripts/compute_norm_stats.py --
        # run once before this config's first training job) -- the expanded 31-task
        # set's joint-position distribution differs from the 4-task sanity subset's
        # (same reasoning as every other *_expanded arm needing fresh norm_stats vs.
        # its small-subset counterpart).
        name="yor_icl_pi05_expanded_absolute_joint",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorAbsoluteJointDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=128,
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
        # Sanity check for a genuinely aligned absolute-joint action/proprioception
        # pair, on Hannibal52Barca/icl-dataset-fixed-action (root built via
        # nyu-finger-robot/tools/merge_hf_aligned_columns.py): state is the usual
        # 15-dim observation.state, action is that dataset's real action.q_target
        # (14-dim, IK-reconstructed from action.left_ee/right_ee and anchored
        # per-episode to observation.state's own FK -- see LeRobotYorAlignedQDataConfig's
        # docstring). Unlike yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k, this is
        # a real recorded action, not observation.state's own future window reused as
        # a pseudo-action -- fixes the action/observation mismatch that config's
        # docstring flags. Same 4-task pnp episode subset/prompt overrides as every
        # other *_sanity15k config (yor_icl_pi05_easy_pnp_v2_episodes.json,
        # YOR_TASK_PROMPT_OVERRIDES), so filters match exactly.
        #
        # 4xH200/batch 256 (64cpu/800G, matching yor_icl_pi05_expanded_frozen_vision's
        # 4xH200 precedent), decay_steps matched to num_train_steps.
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_aligned_q_sanity15k/
        # icl-dataset/norm_stats.json, via scripts/compute_norm_stats.py -- run once
        # before this config's first training job).
        name="yor_icl_pi05_aligned_q_sanity15k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorAlignedQDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-action",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # yor_icl_pi05_aligned_q_sanity15k's aligned absolute-joint action space,
        # scaled up to the full 50k steps on viktr.data.icl_dataset.expanded_tasks()
        # (31 of icl-dataset's 36 tasks, 1,784 episodes) instead of the 4-task/196-
        # episode sanity subset -- same scale-up yor_icl_pi05_expanded_absolute_joint
        # is to yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k.
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_aligned_q_extended/icl-dataset/
        # norm_stats.json, via scripts/compute_norm_stats.py -- run once before this
        # config's first training job) -- the expanded 31-task set's joint-position
        # distribution differs from the 4-task sanity subset's.
        name="yor_icl_pi05_aligned_q_extended",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorAlignedQDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-action",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
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
        # Sanity check for a genuinely aligned delta-ee(+absolute-orientation) action /
        # absolute-ee proprioception pair, on Hannibal52Barca/icl-dataset-fixed-obs
        # (root built via nyu-finger-robot/tools/merge_hf_aligned_columns.py): state is
        # that dataset's new observation.left_ee/right_ee (absolute EE pose, FK'd from
        # the real joint encoders), action is the usual per-arm window-relative delta
        # position + absolute rot6d orientation computed from action.left_ee/right_ee
        # (see LeRobotYorAlignedDeltaRot6DDataConfig's docstring). Unlike
        # pi05_extended_deltarot6d, proprioception is EE-space (matching the action),
        # not joint-space observation.state -- fixes the cross-modality mismatch that
        # config's docstring flags. Same 4-task pnp episode subset/prompt overrides as
        # every other *_sanity15k config (yor_icl_pi05_easy_pnp_v2_episodes.json,
        # YOR_TASK_PROMPT_OVERRIDES), so filters match exactly.
        #
        # 4xH200/batch 256, decay_steps matched to num_train_steps -- same convention
        # as yor_icl_pi05_aligned_q_sanity15k above.
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_aligned_deltarot6d_sanity15k/
        # icl-dataset/norm_stats.json, via scripts/compute_norm_stats.py -- run once
        # before this config's first training job).
        name="yor_icl_pi05_aligned_deltarot6d_sanity15k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorAlignedDeltaRot6DDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # yor_icl_pi05_aligned_deltarot6d_sanity15k's aligned delta-ee/absolute-ee
        # action/proprioception pair, scaled up to the full 50k steps on
        # viktr.data.icl_dataset.expanded_tasks() (31 of icl-dataset's 36 tasks, 1,784
        # episodes) instead of the 4-task/196-episode sanity subset -- same scale-up
        # pattern as every other *_expanded/*_extended arm in this file.
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_aligned_deltarot6d_extended/
        # icl-dataset/norm_stats.json, via scripts/compute_norm_stats.py -- run once
        # before this config's first training job) -- the expanded 31-task set's
        # EE-pose distribution differs from the 4-task sanity subset's.
        name="yor_icl_pi05_aligned_deltarot6d_extended",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorAlignedDeltaRot6DDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
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
        # Arm 1 of the ICRA canonical action/observation space (notes/ICRA_plan.md):
        # proprioception is ONLY absolute EE position + absolute rot6d orientation, per
        # arm (18-dim; no gripper, no lift, no base) -- unlike
        # yor_icl_pi05_aligned_deltarot6d_sanity15k, which uses raw quat+xyz (14-dim)
        # for state. Action is unchanged from that arm: per-arm delta EE position
        # (relative to the query window's own first frame) + absolute rot6d
        # orientation + gripper (20-dim; no lift, no base). See
        # yor_policy.YorInputsCanonical's docstring for the exact derivation.
        #
        # Same root (icl-dataset-fixed-obs) and same 4-task pnp episode subset/prompt
        # overrides as yor_icl_pi05_aligned_deltarot6d_sanity15k
        # (yor_icl_pi05_easy_pnp_v2_episodes.json, YOR_TASK_PROMPT_OVERRIDES), so
        # filters match exactly. 4xH200/batch 256, decay_steps matched to
        # num_train_steps -- same convention as every other *_sanity15k config above.
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_canonical_sanity15k/
        # icl-dataset/norm_stats.json, via scripts/compute_norm_stats.py -- run once
        # before this config's first training job) -- rot6d state is a genuinely
        # different distribution from every existing arm's quat/joint-space state.
        name="yor_icl_pi05_canonical_sanity15k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # yor_icl_pi05_canonical_sanity15k's canonical action/observation space,
        # scaled up on a trimmed subset of viktr.data.icl_dataset.expanded_tasks()
        # (31 of icl-dataset's 36 tasks, 1,784 episodes) instead of the 4-task/196-
        # episode sanity subset -- same scale-up pattern as every other
        # *_expanded/*_extended arm in this file.
        #
        # Unlike every other *_extended arm, this one drops 11 of those 31 tasks
        # (assets/yor_icl_canonical_extended_episodes.json, 1,186 episodes, 20 tasks
        # -- built by filtering yor_icl_expanded_episodes.json on task string):
        # hit_the_yellow_cube_..._left_arm, open_the_gatorade_bottle,
        # pass_the_{red_chilli,salt_shaker,yellow_cube}_from_the_right_arm_to_the_
        # left_arm, pick_up_the_orange_cube_..._{left,right}_arm_on_an_uncluttered_
        # table, put_the_bottle_on_the_plate(_with_the_right_arm), sort_the_items_
        # into_their_containers, stack_the_three_cups_into_a_tower. Those last two
        # alone are ~62% of the removed frame volume (216,171 + 103,079 of 507,846
        # removed frames) despite being only 100 of the 598 removed episodes --
        # by far the longest-horizon tasks in the set, and (per
        # yor_retrieval._load_pool_cached's docstring) sort_the_items_into_their_
        # containers is one of the two largest/most memory-heavy retrieval pools
        # (~5.9GB) -- dropping it here also caps that pressure for any later VICTR
        # arm built on this same trimmed episode set. Net effect: 1,784->1,186
        # episodes (-33.5%), 1,019,948->512,102 frames (-49.8%).
        #
        # Needs its own norm_stats (assets/yor_icl_pi05_canonical_extended/
        # icl-dataset/norm_stats.json, via scripts/compute_norm_stats.py -- run once
        # before this config's first training job, and again any time
        # episodes_path changes) -- the expanded task set's EE-pose distribution
        # differs from the 4-task sanity subset's.
        name="yor_icl_pi05_canonical_extended",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
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
        # ICRA canonical arm 3 (notes/ICRA_plan.md): VICTR vision-metric retrieval on
        # the trimmed 20-task/1,186-episode canonical set (yor_icl_pi05_canonical_
        # extended's episodes_path). Precedent: yor_icl_victr_vision_expanded.
        #
        # pool_dir points at assets/victr_icl_pool_canonical_224 -- a copy of the
        # original 31-task expanded pool (assets/victr_icl_pool_expanded_224) with
        # proprio/actions rebuilt to the canonical representation (pos+rot6d state,
        # delta-from-chunk-start+rot6d+gripper action) for exactly the 20 tasks this
        # episode set can query, so chunk_to_context_text's digitized "State: ...;
        # Action: ..." description matches what the query side now predicts. images/
        # key_embeddings/key_values are byte-identical to the original pool --
        # retrieval selection itself is representation-agnostic (DINOv2 image
        # embeddings only), only the descriptive text changes. The 11 excluded tasks'
        # pools were not copied (never queried by this episode set, so never loaded).
        #
        # precomputed_context_dir is REUSED UNCHANGED from the original expanded set
        # (outputs/victr/retrieval_context_expanded) -- confirmed by reading
        # RetrievalContextInputs.__call__: for a Pi0VictrConfig model (use_action_
        # interpolation always False here), the precomputed context depends only on
        # the query's own image (vision metric) and the pool's images/embeddings --
        # never on the query's action/state encoding. All 1,186 of this episode set's
        # episodes are covered by the existing 1,784-episode-scoped precompute (0
        # missing, confirmed directly). Only RICL (arm 6, live retrieval + action
        # interpolation) actually needs a fresh precompute for canonical.
        #
        # norm_stats reused via AssetsConfig from yor_icl_pi05_canonical_extended --
        # same episodes_path, same state/action encoding (LeRobotYorVictrCanonicalDataConfig
        # shares YorInputsCanonical/YorOutputsCanonical with LeRobotYorCanonicalDataConfig),
        # so recomputing would just reproduce the same numbers.
        #
        # batch_size=256, per explicit request -- every existing VICTR precedent in
        # this file uses 128, not 256 (the retrieval-context arms carry extra
        # per-sample tokens -- context chunks/text -- that the plain pi05/pi0-fast
        # arms don't, so this is untested at 256 for a VICTR arm here; see
        # ICRA_plan.md Sec 2's flag). Real OOM risk on this 47h job if it doesn't fit;
        # not yet validated by a short smoke run before committing all 3 arms to it.
        name="yor_icl_victr_vision_canonical",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision"),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # ICRA canonical arm 4 -- same as yor_icl_victr_vision_canonical, retrieval_metric
        # ="value" instead of "vision" (RoboDopamine progress-value nearest-neighbor,
        # see yor_retrieval.py). Precedent: yor_icl_victr_value_expanded.
        name="yor_icl_victr_value_canonical",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="value"),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="value",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # ICRA canonical arm 5 -- same as yor_icl_victr_vision_canonical, retrieval_metric
        # ="vision_value" (fixed-weight vision+value fusion, paper Sec III-G -- per-query
        # z-score of vision distance and value distance, summed, NOT the paper's learned
        # gpsi). Precedent: yor_icl_victr_vision_value_expanded.
        name="yor_icl_victr_vision_value_canonical",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision_value"),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision_value",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # ICRA canonical arm 6 -- RICL (arXiv:2508.02062; notes/action_interpolation.md):
        # blends the model's own predicted action tokens with the top-1 retrieved
        # chunk's actual action tokens at every decode step, weighted by
        # exp(-lamda * vision distance). Precedent: yor_icl_fast_victr_vision_interp_expanded.
        #
        # fast_tokenizer_path points at the freshly-fit canonical tokenizer
        # (third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical, via
        # scripts/fit_canonical_fast_tokenizer.py -- fit on the canonical action
        # encoding, not the old raw one; see that script's docstring and
        # ICRA_plan.md Sec 3.2). max_action_tokens=224 kept at the existing
        # expanded-set precedent -- the new tokenizer's own measured compression
        # stats (max_token_length=175 over 47,232 chunks) fit comfortably under it,
        # no retuning needed.
        #
        # pool_dir MUST be the canonical-patched pool (assets/victr_icl_pool_canonical_224,
        # see the §1a addendum in ICRA_plan.md) -- unlike arms 3-5, this isn't just
        # representational hygiene for RICL: use_action_interpolation reads
        # `chunk['actions']` directly out of the pool to blend against the model's own
        # prediction (RetrievalContextInputs.__call__), so it must already be in
        # canonical (delta+rot6d+grip) form or the blend is nonsensical.
        #
        # precomputed_context_dir now set: job 17355272 (this arm's first submission)
        # averaged only 15.7% GPU utilization and was killed for it -- live retrieval
        # runs DINOv2 CPU-only in the dataloader (yor_retrieval.load_dinov2) on every
        # sample, which starved the GPU. precompute_retrieval_context.py
        # --use-action-interpolation (added for this arm specifically -- the top-1
        # vision distance + nearest-chunk action tokens action_interpolation_extras()
        # needs are now precomputable too, same DINOv2/pool/task per frame either way)
        # replaces that with an O(1) lookup, same as arms 3-5.
        #
        # context_text_max_length=1024 (up from the Pi0FastVictrConfig default of 64,
        # never overridden in that same first submission): chunk_to_context_text's
        # digitized Task/State/Action text for this pool measures 700-713 tokens
        # (PaligemmaTokenizer), matching 17355272's live truncation warnings
        # (671-702, silently clipped to 64 on ~every batch) -- 1024 leaves headroom.
        #
        # norm_stats reused via AssetsConfig from yor_icl_pi05_canonical_extended, same
        # convention as arms 3-5.
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
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-canonical"
                ),
            },
        ),
        data=LeRobotYorVictrCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_canonical_extended"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_canonical_extended_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_canonical_224",
            # Written by shells/slurm/victr_precompute_canonical_interp_vision_job.sh
            # (precompute_retrieval_context.py --use-action-interpolation --canonical-actions).
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_canonical_interp",
            use_action_interpolation=True,
            lamda=10.0,
            max_action_tokens=224,
        ),
        # batch_size=64, not the arm-6-precedent's 128 or the plain-arms' 256: this
        # config's context_text_max_length=1024 is 16x yor_icl_fast_victr_vision_
        # interp_expanded's default-64 context length (needed for correctness -- see
        # the comment above -- not reducible), and every OOM'd tensor in job
        # 17431526/17390189's traceback (135.36GiB pre-remat, the 98.05GiB single
        # allocation that failed) scales at least linearly with batch_size on top of
        # that already-16x-larger context. 256/4=64 guarantees at least a 4x cut on
        # both of those, well under an H200's ~141GiB -- 128 (only 2x cut) was not
        # tried first because the 4x margin was judged worth the up-front throughput
        # cost, given this is a 47h unattended job.
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
        # ICRA canonical arm 7 -- pi0-FAST analogue of yor_icl_pi05_canonical_extended:
        # same canonical action/observation space + trimmed 20-task episode set, but the
        # discrete FAST-tokenized/autoregressive backend (pi0_fast.Pi0FASTConfig) instead
        # of pi0.5's continuous flow-matching backend. Precedent: yor_icl_pi0_fast_expanded.
        # No retrieval -- plain LeRobotYorCanonicalDataConfig, same as arms 1-2.
        #
        # fast_tokenizer_path / max_token_len: same canonical tokenizer and reasoning as
        # arm 6 above.
        #
        # norm_stats reused via AssetsConfig from yor_icl_pi05_canonical_extended, same
        # convention as arms 3-6.
        name="yor_icl_pi0_fast_canonical",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-canonical"
                ),
            },
        ),
        data=LeRobotYorCanonicalDataConfig(
            repo_id="icl-dataset",
            root="/scratch/lim2045/icl_ws/icl-dataset-fixed-obs",
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
    TrainConfig(
        # Same as yor_icl_pi05_expanded, from-scratch (pi05_base) rerun with the
        # residual quantile-normalization bug fixed: action dims 16:20 (base_vel +
        # lift_cmd -- always ~0, this is stationary tabletop bimanual data, base/lift
        # are never commanded) are dropped entirely (LeRobotYorDataConfig.
        # drop_base_lift=True -> yor_policy.YorInputs/YorOutputs) instead of zeroed.
        #
        # yor_icl_pi05_expanded's own norm_stats.json has near-identical, near-zero
        # q01/q99 for those dims (computed from the raw, not-exactly-zero teleop
        # jitter) -- zeroing the *value* doesn't fix Normalize's
        # (x-q01)/(q99-q01+1e-6) quantile formula, which still turns the deliberately-
        # zeroed x=0 into a huge constant (e.g. ~63 for base_vel.vx, whose q01==q99 to
        # float32 precision). That alone accounts for ~125 of the ~161 step-0 flow-
        # matching loss measured on the original run (wandb run lfa6uo1f, see
        # notes/training_runs.md) -- a real, reproducible artifact an ML reviewer
        # flagged as implausible for a properly normalized init loss, distinct from
        # (and much smaller than) the earlier median-46k/max-6.3e5 bug that was fixed
        # before that run.
        #
        # Dropping the dims (rather than re-deriving new norm_stats) means
        # Normalize._normalize_quantile's `stats.q01[..., :x.shape[-1]]` slicing
        # naturally only ever sees the first 16 (real) entries of the EXISTING
        # yor_icl_pi05_expanded norm_stats.json -- reused via AssetsConfig, no new
        # asset dir needed. State (15-dim) and the model's own action_dim (pi05's
        # native 32, via PadStatesAndActions) are untouched -- padding 16->32 is the
        # same zero-pad-unused-dims convention the model already used for 20->32.
        #
        # Every other hyperparameter is identical to yor_icl_pi05_expanded
        # (same episodes, batch size, schedule, fresh from pi05_base) so the two runs'
        # loss curves are directly comparable.
        name="pi05_extended_quantilesfixed",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            drop_base_lift=True,
        ),
        batch_size=128,
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
        # pi05_extended_quantilesfixed, but with proprioception dropped from 10% of
        # training examples (LeRobotYorDataConfig.state_dropout_prob -> ModelTransform
        # Factory -> _transforms.TokenizePrompt) so the policy can't over-rely on
        # state -- for the other 90% the "State: ..." clause in the discrete prompt is
        # unchanged. Fresh from pi05_base (not a continuation of any existing
        # quantilesfixed checkpoint): those checkpoints were trained with state always
        # present, so resuming from their weights wouldn't isolate the effect of this
        # change from the earlier run's already-learned state-reliance.
        #
        # Every other hyperparameter identical to pi05_extended_quantilesfixed so the
        # two runs' loss curves are directly comparable.
        #
        # ASSUMPTIONS (no clarification requested, per instruction):
        # - "10% of the time" means per-example (per training sample), not per-episode
        #   or per-batch.
        # - "No access to proprioception" means the discrete "State: ..." clause is
        #   omitted from the tokenized prompt entirely for that example (not zeroed) --
        #   pi05's only proprioception input is this discretized state-in-prompt path
        #   (see pi0.py Pi0.embed_suffix: the continuous state_proj token only exists
        #   `if not self.pi05`), so dropping it here removes 100% of the state signal
        #   for the affected examples.
        # - drop_base_lift=True, same batch_size/num_workers/schedule/optimizer as the
        #   base config -- isolating state_dropout_prob as the only changed variable.
        name="pi05_extended_quantilesfixed_state_dropout",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            drop_base_lift=True,
            state_dropout_prob=0.1,
        ),
        batch_size=128,
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
        # Continuation of pi05_extended_quantilesfixed_2xbatch's finished 50k-step run
        # -- checkpoints/pi05_extended_quantilesfixed/pi05_extended_quantilesfixed_2xbatch/
        # 49999/ -- for a further 50k steps. NOT pi05_extended_quantilesfixed_full's
        # checkpoint: _2xbatch is the more recent of the two (batch_size=256/
        # num_workers=16 on 4 GPUs, itself a from-scratch pi05_base run despite the
        # name suggesting otherwise -- see its wandb config.yaml's weight_loader).
        #
        # Mirrors yor_icl_pi05_expanded_continued_50k's pattern (resume weights only,
        # fresh no-warmup schedule over this run's own step span; every other
        # hyperparameter identical to the source run, batch_size/num_workers included).
        #
        # num_train_steps=50_000 is this run's OWN step count (0..50000, not
        # 50000..100000) -- only params are seeded from the source checkpoint,
        # optimizer state/step/wandb id are not carried over.
        #
        # lr_schedule is a user-specified narrower anneal (5e-6 -> 2.5e-6, no warmup)
        # rather than restarting the original 5e-5 peak -- continuing at the tail end
        # of the source run's own decayed LR, annealing further down from there.
        name="pi05_extended_quantilesfixed_continued_50k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            drop_base_lift=True,
        ),
        batch_size=256,
        num_workers=16,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-6,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_extended_quantilesfixed/pi05_extended_quantilesfixed_2xbatch/49999/params"
        ),
    ),
    TrainConfig(
        # Same as pi05_extended_quantilesfixed_continued_50k, but continuing
        # pi05_extended_quantilesfixed_full's checkpoint (batch_size=128/num_workers=8,
        # 2 GPUs) instead of _2xbatch's -- a parallel continuation of the OTHER
        # finished quantilesfixed run, submitted alongside the _2xbatch continuation
        # specifically to occupy the reservation's remaining GPU capacity (2 GPUs here
        # vs. that config's 4, both fit on the same reserved node at once).
        #
        # Same 5e-6 -> 2.5e-6 no-warmup anneal, own 50_000-step span, weights-only
        # resume -- see that config's comment for the full rationale.
        name="pi05_extended_quantilesfixed_full_continued_50k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            drop_base_lift=True,
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-6,
            decay_steps=50_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_extended_quantilesfixed/full/49999/params"
        ),
    ),
    TrainConfig(
        # Same episodes/schedule/batch-size/fresh-from-pi05_base as yor_icl_pi05_expanded
        # -- only the action representation differs, per a specialist's suggestion
        # (chat 2026-08-31): per arm, delta cartesian position (relative to this
        # window's own first frame -- see YorInputsDeltaRot6D) instead of absolute xyz,
        # and orientation as the first two rows of the rotation matrix (Zhou et al. 2019
        # 6D continuous rotation representation) instead of a quaternion -- quaternions
        # are a double cover of SO(3) (q and -q are the same rotation), a discontinuity
        # that can hand a flow-matching regression loss two visually-adjacent rotations
        # mapped to distant targets. base_vel/lift are dropped entirely, same rationale
        # as pi05_extended_quantilesfixed. See yor_rotation.py/yor_policy.py docstrings.
        #
        # Needs its own norm_stats (assets/pi05_extended_deltarot6d/icl-dataset/
        # norm_stats.json, via scripts/compute_norm_stats.py -- run once before this
        # config's first training job) -- delta positions and 6D rotation values are a
        # genuinely different distribution from the plain arms' absolute xyz/quaternion,
        # can't reuse yor_icl_pi05_expanded's the way pi05_extended_quantilesfixed does.
        name="pi05_extended_deltarot6d",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDeltaRot6DDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=128,
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
        # Continuation of yor_icl_pi05_expanded's finished 50k-step run --
        # checkpoints/yor_icl_pi05_expanded/yor_icl_pi05_expanded_full/49999/ -- for a
        # further 50k steps. Mirrors yor_icl_pi05_sanity15k_continued_15k's pattern
        # (resume weights only, fresh no-warmup cosine decay over this run's own step
        # span) rather than yor_icl_pi05_expanded_100k's from-scratch/bigger-batch
        # redo of a 100k-step target starting from pi05_base -- this arm keeps every
        # other hyperparameter (data, batch_size, optimizer) identical to
        # yor_icl_pi05_expanded so the two 50k halves are directly, apples-to-apples
        # comparable, unlike the _100k arm (different batch size and schedule).
        #
        # num_train_steps=50_000 is this run's OWN step count (0..50000, not
        # 50000..100000) -- only params are seeded from the source checkpoint,
        # optimizer state/step/wandb id are not carried over (same convention as
        # yor_icl_pi05_sanity15k_continued_15k).
        #
        # Reuses yor_icl_pi05_expanded's already-computed norm_stats via AssetsConfig
        # (same convention as yor_icl_pi05_expanded_100k) instead of copying the json.
        name="yor_icl_pi05_expanded_continued_50k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/yor_icl_pi05_expanded/yor_icl_pi05_expanded_full/49999/params"
        ),
    ),
    TrainConfig(
        # pi0-FAST analogue of yor_icl_pi05_expanded above -- same expanded 31-task,
        # 1,784-episode data, but the discrete FAST-tokenized/autoregressive backend
        # (pi0_fast.Pi0FASTConfig) instead of pi0.5's continuous flow-matching backend.
        # This is the "extended full run" companion to
        # yor_icl_pi0_fast_easy_pnp_v2_sanity15k_annealing's cheap 4-task/15k-step
        # sanity check -- run the sanity check first to validate the FAST-tokenized
        # pipeline before committing to this arm's full 50k-step budget on the bigger
        # dataset.
        #
        # action_dim=20/action_horizon=30 match icl-dataset's actual action shape (same
        # as every other yor_icl_* arm -- see yor_policy.YorInputs); max_token_len=256
        # matches yor_icl_fast_victr_vision_expanded's choice for the identical action
        # shape and data (that config's comment found action-only FAST tokens can reach
        # ~191). fast_tokenizer_path reuses yor_icl_ki_expanded_subtask's /
        # yor_icl_fast_victr_vision_expanded's already-fit expanded-set tokenizer
        # (outputs/fast_tokenizer/yor-icl-expanded, fit on this SAME 31-task action
        # distribution via scripts/fit_expanded_fast_tokenizer.py) -- no new tokenizer
        # fit needed.
        #
        # Reuses yor_icl_pi05_expanded's norm_stats.json (copied to
        # assets/yor_icl_pi0_fast_expanded/icl-dataset/norm_stats.json -- same episodes,
        # so identical state/action distributions; same convention as the other
        # expanded-set arms in this file).
        name="yor_icl_pi0_fast_expanded",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-expanded"
                ),
            },
        ),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=128,
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
    TrainConfig(
        # Continuation of yor_icl_pi0_fast_expanded's finished 50k-step run --
        # checkpoints/yor_icl_pi0_fast_expanded/full/49999/ -- for a further 50k steps.
        # Mirrors yor_icl_pi05_expanded_continued_50k's pattern exactly (resume params
        # only, fresh no-warmup cosine decay over this run's own 0..50000 step span,
        # every other hyperparameter identical to the source run) -- same rationale,
        # just for the FAST/discrete arm instead of the flow-matching one. Optimizer
        # state/step/wandb id are not carried over.
        #
        # Reuses yor_icl_pi0_fast_expanded's own norm_stats via AssetsConfig instead of
        # copying the json (that run's own norm_stats.json was itself already a copy of
        # yor_icl_pi05_expanded's, per that config's comment above).
        name="yor_icl_pi0_fast_expanded_continued_50k",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-expanded"
                ),
            },
        ),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi0_fast_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=50_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/yor_icl_pi0_fast_expanded/full/49999/params"
        ),
    ),
    TrainConfig(
        # Same as yor_icl_pi05_expanded (same expanded 1,784-episode data, same fixed
        # action dims 16:20 -- see yor_policy.YorInputs), but with the SigLip vision
        # tower (PaliGemma.img, see pi0.py's `self.PaliGemma = nnx.Dict(llm=llm,
        # img=img)`) frozen -- Pi0Config.get_freeze_filter() only covers LoRA, so this
        # needs its own explicit freeze_filter. Gemma LM + action expert (PaliGemma.llm)
        # stay fully trainable. Scaled to 4 H200s / global batch 256 (same 64/device as
        # the 2-GPU/128-batch baseline) and a longer 70k-step run, with decay_steps
        # matched to num_train_steps so the cosine schedule actually bottoms out at
        # decay_lr by the end (rather than sitting flat at the floor for the last steps,
        # per the yor_icl_ki_subtask_rabc annealing discussion).
        name="yor_icl_pi05_expanded_frozen_vision",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        freeze_filter=nnx_utils.PathRegex(".*img.*"),
        batch_size=256,
        num_workers=8,
        num_train_steps=70_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=70_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Same as yor_icl_pi05_expanded_frozen_vision above, but data.root points at a
        # pre-decoded, pre-resized (224x224) image-mode copy of the same 1,784 episodes
        # (third_party/openpi/scripts/precompute_frame_cache.py), instead of the shared
        # icl-dataset's live-decoded video. Job 16476252 (this config without caching)
        # was killed by the cluster's <80% GPU-util reaper -- freezing the vision
        # backbone cuts backward-pass compute per step, but live torchcodec/pyav decode
        # of 3 cameras/sample stayed fixed, so the GPU sat idle waiting on the
        # dataloader (avg 59% util measured, see logs/gpu-util-16476252.csv). The cache
        # is bit-identical to the live path (same resize_with_pad/_parse_image calls,
        # applied once instead of once per epoch -- see that script's docstring), so
        # this trains on the same data as the *_full run, just without the IO stall.
        # episodes_path is range(1784) (not yor_icl_expanded_episodes.json's original
        # indices): the cache is a fresh, reindexed dataset containing only those
        # episodes, renumbered 0..1783 by construction.
        # Reuses yor_icl_pi05_expanded_frozen_vision's already-computed norm_stats
        # (state/action distributions are untouched -- only images were re-encoded)
        # instead of recomputing them, via the assets_dir override below.
        name="yor_icl_pi05_expanded_frozen_vision_cached",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded_frozen_vision"),
            base_config=DataConfig(prompt_from_task=True),
            root="assets/yor_icl_pi05_expanded_frozen_vision_cache_224",
            episodes_path="assets/yor_icl_expanded_frozen_vision_cache_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        freeze_filter=nnx_utils.PathRegex(".*img.*"),
        batch_size=256,
        num_workers=8,
        num_train_steps=70_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=70_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Same as yor_icl_pi05_expanded (same expanded 1,784-episode data, same fixed
        # action dims 16:20), full fine-tune (vision NOT frozen -- no freeze_filter,
        # unlike yor_icl_pi05_expanded_frozen_vision above), scaled to 4 H200s / global
        # batch 256 (same 64/device as the 2-GPU/128-batch baseline) and a longer
        # 100k-step run, decay_steps matched to num_train_steps so the cosine schedule
        # bottoms out at decay_lr by the end (same reasoning as the frozen-vision sibling
        # above). Unfrozen vision keeps backward-pass compute per step high enough that
        # this shouldn't hit the frozen arm's <80%-util dataloader-starvation failure
        # (job 16476252 -- see yor_icl_pi05_expanded_frozen_vision_cached above); if it
        # does, switch to the *_cached-style precomputed frame cache instead of live
        # torchcodec/pyav decode. Reuses yor_icl_pi05_expanded's already-computed
        # norm_stats (identical data/action space) instead of recomputing them.
        name="yor_icl_pi05_expanded_100k",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYorDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
        ),
        batch_size=256,
        num_workers=8,
        num_train_steps=100_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # VICTR (retrieval-conditioned IC-VLA, paper Sec III-E) "vision" arm: same 4-task
        # icl-dataset pick-and-place data/hyperparameters as yor_icl_pi05_easy_pnp_v2, but
        # each query is conditioned on num_context_chunks demonstration chunks retrieved
        # by DINOv2 visual similarity (openpi.policies.yor_retrieval.vision_retrieve).
        # Cross-checks lerobot's own "vision" VICTR arm (viktr/policy/pi05_context.py,
        # scripts/train_victr.py) on openpi's independently-implemented pi05 backbone.
        name="yor_icl_victr_vision",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # VICTR "value" arm: same as yor_icl_victr_vision, but context chunks are
        # retrieved by RoboDopamine progress-value alignment instead of visual similarity
        # (openpi.policies.yor_retrieval.value_retrieve). Cross-checks lerobot's "value"
        # VICTR arm (currently running as SLURM job 16082412).
        name="yor_icl_victr_value",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="value"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="value",
        ),
        batch_size=128,
        num_workers=8,
        num_train_steps=15_000,
        seed=42,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=15_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-6, weight_decay=0.01, clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=5_000,
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Same as yor_icl_victr_value, but trained on the 31-task expanded set (see
        # yor_icl_pi05_expanded) with retrieval context precomputed offline instead of
        # live -- pool_dir points at the 224x224-resized pool (assets/
        # victr_icl_pool_expanded_224, see scripts/resize_pool_images.py) and
        # precomputed_context_dir at scripts/precompute_retrieval_context.py's
        # completed output (outputs/victr/retrieval_context_expanded/value, all
        # 1,784 episodes -- confirmed complete before this config was added). See
        # RetrievalContextInputs.__call__: when precomputed_context_dir is set, this
        # is an O(1) per-episode array lookup, no live DINOv2/pool-search/tokenize
        # per sample -- the fix for the sub-80%-GPU-utilization issue that motivated
        # precomputing in the first place.
        #
        # norm_stats.json is copied from yor_icl_pi05_expanded's, not computed fresh --
        # same episodes_path (same 1,784-episode expanded set) means identical state/
        # action distributions, so recomputing would just reproduce the same numbers at
        # the cost of another full video-decode pass. Same convention already
        # established for yor_icl_ki_subtask/yor_icl_ki_expanded_subtask (see those
        # configs).
        name="yor_icl_victr_value_expanded",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="value"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="value",
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
        ),
        batch_size=128,
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Fixed-weight vision+value fusion arm (paper Sec III-G) -- per-query z-score
        # of vision distance and value distance, summed, NOT the paper's learned gpsi
        # trained jointly with the IC-VLA (see openpi.policies.yor_retrieval's module
        # docstring and the "Add a vision+value fusion VICTR arm" plan for why).
        # Shares the same precomputed_context_dir as every other expanded-set VICTR
        # arm, reading from its own vision_value/ subdirectory (all 1,784 episodes).
        name="yor_icl_victr_vision_value_expanded",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision_value"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision_value",
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
        ),
        batch_size=128,
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Same as yor_icl_victr_value_expanded, but retrieval_metric="vision" instead
        # of "value" -- the plain-pi05-backend counterpart to yor_icl_fast_victr_
        # vision_expanded (whose own comment already claimed to reuse this config's
        # precomputed context; that claim only became true once this config was
        # actually added).
        # Shares the same precomputed_context_dir as every other expanded-set VICTR
        # arm (outputs/victr/retrieval_context_expanded/vision, all 1,784 episodes).
        #
        # norm_stats.json copied from yor_icl_pi05_expanded's, same convention as the
        # other expanded arms (identical episodes_path -> identical state/action
        # distribution, no need to recompute).
        name="yor_icl_victr_vision_expanded",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
        ),
        batch_size=128,
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Full-scale vision arm with both fixes applied: drop_base_lift=True (the
        # residual quantile-norm fix from pi05_extended_quantilesfixed -- see
        # yor_policy.YorInputs' docstring) AND a 90/10 train/pool split with no test
        # holdout (build_retrieval_pool.py --train-frac 0.9 --pool-frac 0.1
        # --test-frac 0.0), instead of yor_icl_victr_vision_expanded's 85/10/5 whose
        # openpi episodes_path (assets/yor_icl_expanded_episodes.json) is the full
        # train+pool+test union -- confirmed by direct set comparison against
        # outputs/victr/icl_pool_expanded/splits.json that ~10% of THAT config's
        # training queries come from episodes also embedded in its own retrieval
        # pool, so retrieve_chunks/_topk (no self-episode exclusion) can hand a query
        # back chunks from its own episode. Here episodes_path points at the new
        # split's train_episodes.json (train-only, zero overlap with pool_dir's
        # episodes by construction), closing that leak entirely rather than filtering
        # it post-hoc out of the old split.
        #
        # norm_stats.json reused via AssetsConfig from yor_icl_pi05_expanded, same
        # convention as pi05_extended_quantilesfixed -- drop_base_lift slices actions
        # to 16 dims before Normalize ever sees the old 20-dim stats' last 4 entries.
        #
        # Needs its own pool/precomputed-context assets (new split -> different pool
        # membership -> different nearest-neighbor results for every query, so the
        # old precomputed_context_expanded can't be reused) -- see notes/
        # training_runs.md for the exact build_retrieval_pool.py / precompute_
        # retrieval_context.py invocations used to produce them before this config's
        # first training job.
        name="viktr_vision_50k",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="vision"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_expanded_90_10_train_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_90_10_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded_90_10",
            drop_base_lift=True,
        ),
        batch_size=128,
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Same as viktr_vision_50k, but retrieval_metric="value" -- see that config's
        # comment for the drop_base_lift + 90/10-split rationale shared by both.
        name="viktr_value_50k",
        model=pi0_victr.Pi0VictrConfig(pi05=True, action_horizon=30, retrieval_metric="value"),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            assets=AssetsConfig(assets_dir="assets/yor_icl_pi05_expanded"),
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="value",
            episodes_path="assets/yor_icl_expanded_90_10_train_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_90_10_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded_90_10",
            drop_base_lift=True,
        ),
        batch_size=128,
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
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    TrainConfig(
        # Native RICL-equivalent baseline: VICTR's vision-metric context conditioning
        # (openpi.models.pi0_fast_victr.Pi0FastVictr) ported onto the discrete,
        # autoregressive pi0-FAST backend instead of pi0.5's continuous flow-matching
        # backend (see that module's docstring for the port). Reuses the SAME
        # precomputed vision-metric retrieval context as yor_icl_victr_vision_expanded
        # (outputs/victr/retrieval_context_expanded/vision) -- no new preprocessing.
        #
        # This is deliberately NOT built on third_party/ricl_openpi (RICL's own DROID-
        # specific fork): that pipeline hardcodes DROID's raw h5+JPG preprocessing, 3
        # fixed camera names, an 8-dim action space, and a separate autofaiss retrieval
        # step, none of which line up with icl-dataset (LeRobot-format, 3 different
        # cameras, 20-dim action). Building on our own DataConfig/ModelTransformFactory
        # pipeline instead reuses almost everything already working for VICTR.
        #
        # fast_tokenizer_path reuses yor_icl_ki_expanded_subtask's already-fit
        # expanded-set tokenizer (fit on the SAME 20-dim/30-horizon action
        # distribution) -- no new tokenizer fit needed. max_token_len=256: that KI
        # config's own measurement found the action-only FAST-token budget alone can
        # reach ~191 tokens for this action shape (max_action_tokens=192 there); this
        # config's max_token_len also has to cover the language prompt text in the
        # SAME budget (unlike KI's separate max_action_tokens), so 256 leaves ~60
        # tokens of headroom over the known action-only max for prompt text -- NOT yet
        # empirically verified against this config's actual tokenized samples (KI's
        # measurement was for its own auxiliary FAST branch, not this arm's primary
        # token stream). If training fails on a token-length assertion or an
        # unexpectedly-truncated action, that's the first thing to check.
        name="yor_icl_fast_victr_vision_expanded",
        model=pi0_fast_victr.Pi0FastVictrConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            num_context_chunks=1,
            context_frames_per_chunk=1,
            retrieval_metric="vision",
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-expanded"
                ),
            },
        ),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_224",
            precomputed_context_dir="/scratch/lim2045/icl_ws/viktr/outputs/victr/retrieval_context_expanded",
        ),
        batch_size=128,
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
        # Generic pi0-FAST base checkpoint (same choice pi0_fast_libero makes) --
        # NOT pi0_fast_droid, our embodiment/action-space isn't DROID's.
        # VictrCheckpointWeightLoader is model-class-agnostic (matches by param path
        # regex, not isinstance), so it's reused unchanged from the pi0.5-backed VICTR
        # arms -- it already lets neighbor_rank_embedding fall back to random init.
        weight_loader=weight_loaders.VictrCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    ),
    TrainConfig(
        # SERVING-ONLY twin of yor_icl_fast_victr_vision_expanded (scripts/serve_policy.py
        # policy:checkpoint --policy.config=yor_icl_fast_victr_vision_expanded_serve
        # --policy.dir=checkpoints/yor_icl_fast_victr_vision_expanded/<exp>/<step>).
        # Same model/checkpoint -- only precomputed_context_dir differs (unset here).
        # policy_config.create_trained_policy() builds the data pipeline straight from
        # a TrainConfig's `data` field, and the training config's precomputed_context_dir
        # makes yor_retrieval.RetrievalContextInputs do an O(1) lookup keyed on
        # (episode_index, frame_index) -- fields that only exist on a replayed dataset
        # row, never on a live robot observation. Point servable checkpoints at THIS
        # config instead: precomputed_context_dir=None makes RetrievalContextInputs fall
        # back to its live path (DINOv2-embed the current camera frame, then
        # nearest-neighbor search assets/victr_icl_pool_expanded_224/<task_slug>.pkl),
        # which only needs prompt + the primary camera image -- both present on a real
        # observation. See notes/deploy_fast_victr_vision.md.
        name="yor_icl_fast_victr_vision_expanded_serve",
        model=pi0_fast_victr.Pi0FastVictrConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            num_context_chunks=1,
            context_frames_per_chunk=1,
            retrieval_metric="vision",
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-expanded"
                ),
            },
        ),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_224",
            # precomputed_context_dir intentionally omitted (None) -- see comment above.
        ),
        batch_size=128,
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
        # RICL action-interpolation baseline (arXiv:2508.02062; notes/action_interpolation.md):
        # yor_icl_fast_victr_vision_expanded + blending the model's own predicted action
        # tokens with the top-1 retrieved chunk's actual action tokens at every decode
        # step, weighted by exp(-lamda * vision distance) -- the piece of RICL's actual
        # method our native FAST-victr port (built on our own retrieval/data pipeline,
        # not third_party/ricl_openpi's DROID-specific fork) hadn't reproduced yet.
        # Same episodes/tokenizer/pool as yor_icl_fast_victr_vision_expanded --
        # precomputed_context_dir must stay unset here (use_action_interpolation needs
        # the live top-1 distance, which the precomputed path doesn't carry).
        # max_action_tokens=224: yor_icl_fast_victr_vision_expanded's own config comment
        # measured this action space's FAST-token budget can reach ~191 tokens; 224
        # leaves headroom without the KI arm's separate re-measurement (that arm's 64
        # was fit to a different, 4-task-only tokenizer).
        name="yor_icl_fast_victr_vision_interp_expanded",
        model=pi0_fast_victr.Pi0FastVictrConfig(
            action_dim=20,
            action_horizon=30,
            max_token_len=256,
            num_context_chunks=1,
            context_frames_per_chunk=1,
            retrieval_metric="vision",
            use_action_interpolation=True,
            lamda=10.0,
            max_action_tokens=224,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": (
                    "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                    "yor-icl-expanded"
                ),
            },
        ),
        data=LeRobotYorVictrDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            retrieval_metric="vision",
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            pool_dir="assets/victr_icl_pool_expanded_224",
            # precomputed_context_dir intentionally omitted (None) -- live retrieval
            # required, same reasoning as the _serve config above.
            use_action_interpolation=True,
            lamda=10.0,
            max_action_tokens=224,
        ),
        batch_size=128,
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
        # Knowledge Insulation + subtask prediction: flow-matching loss's gradient is
        # kept off the PaliGemma backbone, which instead keeps training via discrete
        # next-token CE losses on FAST-tokenized actions and the active subtask's text
        # (openpi.models.pi0_ki.Pi0Ki). Cross-checks lerobot's own KI+subtask arm
        # (nyu-finger-robot/configs/yor-pi05-ablation-ki-subtask.yaml). Same 4-task
        # icl-dataset episodes as the other arms, but QUANTILES normalization (see
        # LeRobotYorKiDataConfig's docstring).
        #
        # max_action_tokens=64 (not lerobot's 256, fit for the full 36-task population's
        # up-to-191-token budget): this 4-task subset's actual FAST-tokenized action
        # chunks measured only 24-36 tokens (see openpi.policies.yor_ki's data-pipeline
        # verification), so 256 was mostly wasted padding -- and each padded position
        # gets forwarded through Pi0Ki's discrete-loss branch AND projected through the
        # 257k-token vocab LM head, so the padding was materially contributing to an
        # OOM even on 2xH200 at this arm's batch_size=128. 64 keeps ~2x headroom over
        # the observed max.
        name="yor_icl_ki_subtask",
        model=pi0_ki.Pi0KiConfig(
            pi05=True,
            action_horizon=30,
            fast_tokenizer_path=(
                "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                "yor-icl-pi05-easy-pnp-v2"
            ),
            ki_loss_weight=1.0,
            max_action_tokens=64,
            use_subtask_prediction=True,
            subtask_annotations_path="/scratch/lim2045/icl_ws/icl-dataset/meta/subtask_annotations/subtasks.jsonl",
            subtask_loss_weight=1.0,
        ),
        data=LeRobotYorKiDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            max_action_tokens=64,
        ),
        batch_size=128,
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
        # RA-BC (notes/reward_aligned_bc.md) + Knowledge Insulation/subtask-prediction
        # (openpi.models.pi0_ki.Pi0Ki), combined -- isolates "does RA-BC's per-item
        # loss reweighting help on top of KI" the same way yor_icl_pi05_rabc isolates
        # RA-BC's effect on plain pi05 (see that config's comment). Identical to
        # yor_icl_ki_subtask (same 196-episode 4-task pnp subset, same non-expanded
        # FAST tokenizer, same max_action_tokens=64) with rabc_weights_dir added --
        # reuses yor_icl_pi05_rabc's precomputed weights unchanged, since they're keyed
        # by episode_index over that exact same episode set with the same
        # action_horizon=30 delta.
        #
        # See LeRobotYorKiDataConfig.create()'s comment for how the two compose: RABC
        # reweights only the per-item flow-matching loss; KI's ki_loss/subtask_loss
        # stay batch-uniform (unweighted), since scripts/train.py's loss_fn RABC-
        # weights per_item_loss, and Pi0Ki.compute_loss's aux_loss is already a single
        # scalar broadcast equally onto every item before that weighting happens.
        name="yor_icl_ki_subtask_rabc",
        model=pi0_ki.Pi0KiConfig(
            pi05=True,
            action_horizon=30,
            fast_tokenizer_path=(
                "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/"
                "yor-icl-pi05-easy-pnp-v2"
            ),
            ki_loss_weight=1.0,
            max_action_tokens=64,
            use_subtask_prediction=True,
            subtask_annotations_path="/scratch/lim2045/icl_ws/icl-dataset/meta/subtask_annotations/subtasks.jsonl",
            subtask_loss_weight=1.0,
        ),
        data=LeRobotYorKiDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            max_action_tokens=64,
            rabc_weights_dir="assets/rabc_weights/yor_icl_pi05_rabc",
        ),
        batch_size=128,
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
        # Same as yor_icl_ki_subtask, but trained on the 31-task expanded set (see
        # yor_icl_pi05_expanded) with its own FAST tokenizer -- the original tokenizer
        # was fit only on the 4-task pick-and-place subset's action distribution and
        # would misquantize the expanded set's much more diverse actions (bimanual
        # passing, mallet strikes, sorting, stacking, uncapping, ...); see
        # scripts/fit_expanded_fast_tokenizer.py.
        #
        # max_action_tokens=192: fit_expanded_fast_tokenizer's compression_stats (10%
        # sample of the expanded set's chunks) measured max_token_length=117,
        # p99=61 -- but yor_icl_ki_subtask's own comment records a prior full
        # (36-task) population measurement of up to 191 tokens, which this 31-task
        # expanded set is close enough to that its unsampled true max plausibly
        # exceeds our 117 sample max. 192 covers that known 191 figure with a hair of
        # margin while staying well below 256, which OOM'd at this same
        # batch_size=128 on 2xH200 for yor_icl_ki_subtask (see that config's comment).
        name="yor_icl_ki_expanded_subtask",
        model=pi0_ki.Pi0KiConfig(
            pi05=True,
            action_horizon=30,
            fast_tokenizer_path=(
                "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-expanded"
            ),
            ki_loss_weight=1.0,
            max_action_tokens=192,
            use_subtask_prediction=True,
            subtask_annotations_path="/scratch/lim2045/icl_ws/icl-dataset/meta/subtask_annotations/subtasks.jsonl",
            subtask_loss_weight=1.0,
        ),
        data=LeRobotYorKiDataConfig(
            repo_id="icl-dataset",
            base_config=DataConfig(prompt_from_task=True),
            episodes_path="assets/yor_icl_expanded_episodes.json",
            task_prompt_overrides=YOR_EXPANDED_TASK_PROMPT_OVERRIDES,
            fast_tokenizer_path=(
                "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-expanded"
            ),
            max_action_tokens=192,
        ),
        batch_size=128,
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
