import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.moe as moe
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_moe_config as pi0_moe_config
import openpi.models.pi0_residual_moe_config as pi0_residual_moe_config
import openpi.models.pi05_moe_config as pi05_moe_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as training_config
import openpi.training.moe_weight_loader as moe_weight_loader
import openpi.training.weight_loaders as weight_loaders


def test_pi0_moe_config_defaults():
    config = pi0_moe_config.Pi0MoEConfig()
    assert config.pi05 is False
    assert config.discrete_state_input is False
    assert config.moe_layers is None
    assert config.resolve_moe_layers(4) == ()


def test_pi0_moe_all_layers_sentinel():
    config = pi0_moe_config.Pi0MoEConfig(moe_layers="All")
    assert config.resolve_moe_layers(4) == (0, 1, 2, 3)


def test_moe_feedforward_full_experts_use_nonzero_router_init():
    config = moe.MoEConfig(num_experts=4, top_k=1, router_init_std=1e-3)
    module = moe.MoEFeedForward(features=8, hidden_dim=16, moe_config=config)
    params = module.init(jax.random.key(0), jnp.ones((2, 3, 8), dtype=jnp.float32))["params"]

    assert "gating_einsum_lora_a" not in params["expert_0"]
    assert params["expert_0"]["gating_einsum"].shape == (2, 8, 16)
    assert params["expert_0"]["linear"].shape == (16, 8)
    assert params["router"]["kernel"].shape == (8, 4)
    assert not np.allclose(np.asarray(params["router"]["kernel"]), 0.0)


def test_pi0_libero_moe_configs_use_full_ffn_experts():
    configs = {config.name: config for config in training_config._CONFIGS}  # noqa: SLF001
    for name in ("pi0_libero_moe", "pi0_libero_moe_spatial_only"):
        config = configs[name]
        model_config = config.model

        assert isinstance(model_config, pi0_moe_config.Pi0MoEConfig)
        assert model_config.action_expert_variant == "gemma_300m"
        assert model_config.moe_config.top_k == 1
        assert model_config.moe_config.router_z_loss_coeff == 0.0
        assert model_config.moe_config.load_balance_loss_weight == 0.0
        assert model_config.moe_config.router_init_std == 1e-3
        assert isinstance(config.freeze_filter, nnx.Nothing)


def test_pi05_moe_config_defaults():
    config = pi05_moe_config.Pi05MoEConfig()
    assert config.pi05 is True
    assert config.discrete_state_input is True


def test_pi0_residual_moe_config_defaults():
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig()
    assert config.pi05 is False
    assert config.discrete_state_input is False
    assert config.moe_layers is None
    assert config.resolve_moe_layers(4) == ()
    assert config.moe_config.top_k == 1
    assert config.moe_config.num_experts == 8


def test_pi0_residual_moe_all_layers_sentinel():
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig(moe_layers="All")
    assert config.resolve_moe_layers(4) == (0, 1, 2, 3)


def test_pi0_moe_model_dummy():
    key = jax.random.key(0)
    config = pi0_moe_config.Pi0MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_config=moe.MoEConfig(num_experts=1, top_k=1, router_z_loss_coeff=1e-3),
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    metric_loss, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)
    assert metric_loss.shape == (batch_size, config.action_horizon)
    assert metrics["expert_usage"].shape[1] == config.moe_config.num_experts
    assert metrics["router_entropy"].shape == metrics["expert_usage"].shape[:1]
    assert metrics["load_balance_loss"].shape == metrics["expert_usage"].shape[:1]

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    suffix_tokens, _, _, adarms_cond = model.embed_suffix(obs, act, jnp.ones((batch_size,)))
    assert suffix_tokens.shape[1] == config.action_horizon + 1
    assert adarms_cond is None


def test_pi0_moe_subset_layers_only_report_active_metrics():
    key = jax.random.key(0)
    config = pi0_moe_config.Pi0MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[1, 3],
        moe_config=moe.MoEConfig(num_experts=2, top_k=1, router_z_loss_coeff=1e-3),
    )
    model = config.create(key)

    obs, act = config.fake_obs(2), config.fake_act(2)
    _, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)

    assert metrics["expert_usage"].shape == (2, config.moe_config.num_experts)
    assert metrics["router_entropy"].shape == (2,)
    assert metrics["load_balance_loss"].shape == (2,)


def test_pi0_moe_no_active_layers_reports_empty_metrics():
    key = jax.random.key(0)
    config = pi0_moe_config.Pi0MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[],
        moe_config=moe.MoEConfig(num_experts=2, top_k=1, router_z_loss_coeff=1e-3),
    )
    model = config.create(key)

    obs, act = config.fake_obs(2), config.fake_act(2)
    loss, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)

    assert loss.shape == (2, config.action_horizon)
    assert metrics["expert_usage"].shape == (0, config.moe_config.num_experts)
    assert metrics["router_entropy"].shape == (0,)
    assert metrics["load_balance_loss"].shape == (0,)


def test_pi0_moe_invalid_layer_selection_raises():
    config = pi0_moe_config.Pi0MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[4],
    )

    with np.testing.assert_raises_regex(ValueError, "moe_layers must be between 0 and 3"):
        config.create(jax.random.key(0))


def test_pi05_moe_model_dummy():
    key = jax.random.key(0)
    config = pi05_moe_config.Pi05MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_config=moe.MoEConfig(num_experts=1, top_k=1, router_z_loss_coeff=1e-3),
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    metric_loss, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)
    assert metric_loss.shape == (batch_size, config.action_horizon)
    assert metrics["expert_usage"].shape[1] == config.moe_config.num_experts
    assert metrics["router_entropy"].shape == metrics["expert_usage"].shape[:1]
    assert metrics["load_balance_loss"].shape == metrics["expert_usage"].shape[:1]

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    suffix_tokens, _, _, adarms_cond = model.embed_suffix(obs, act, jnp.ones((batch_size,)))
    assert suffix_tokens.shape[1] == config.action_horizon
    assert adarms_cond.shape == (batch_size, 64)


def test_pi0_residual_moe_model_dummy():
    key = jax.random.key(0)
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers="All",
        moe_config=moe.ResidualMoEConfig(num_experts=2, expert_hidden_dim=16, residual_scale=1.0),
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    metric_loss, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)
    assert metric_loss.shape == (batch_size, config.action_horizon)
    assert metrics["expert_usage"].shape[0] == 4
    assert metrics["expert_usage"].shape[1] == config.moe_config.num_experts
    assert metrics["router_entropy"].shape == metrics["expert_usage"].shape[:1]
    assert metrics["load_balance_loss"].shape == metrics["expert_usage"].shape[:1]

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    suffix_tokens, _, _, adarms_cond = model.embed_suffix(obs, act, jnp.ones((batch_size,)))
    assert suffix_tokens.shape[1] == config.action_horizon + 1
    assert adarms_cond is None


def test_pi0_residual_moe_subset_layers_only_report_active_metrics():
    key = jax.random.key(0)
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[1, 3],
        moe_config=moe.ResidualMoEConfig(num_experts=2, expert_hidden_dim=16, residual_scale=1.0),
    )
    model = config.create(key)

    obs, act = config.fake_obs(2), config.fake_act(2)
    _, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)

    assert metrics["expert_usage"].shape == (2, config.moe_config.num_experts)
    assert metrics["router_entropy"].shape == (2,)
    assert metrics["load_balance_loss"].shape == (2,)


def test_pi0_residual_moe_no_active_layers_reports_empty_metrics():
    key = jax.random.key(0)
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[],
        moe_config=moe.ResidualMoEConfig(num_experts=2, expert_hidden_dim=16, residual_scale=1.0),
    )
    model = config.create(key)

    obs, act = config.fake_obs(2), config.fake_act(2)
    loss, metrics = nnx_utils.module_jit(model.compute_loss_with_moe_metrics)(key, obs, act)

    assert loss.shape == (2, config.action_horizon)
    assert metrics["expert_usage"].shape == (0, config.moe_config.num_experts)
    assert metrics["router_entropy"].shape == (0,)
    assert metrics["load_balance_loss"].shape == (0,)


def test_pi0_residual_moe_invalid_layer_selection_raises():
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[4],
    )

    with np.testing.assert_raises_regex(ValueError, "moe_layers must be between 0 and 3"):
        config.create(jax.random.key(0))


def test_moe_weight_loader_fans_out_dense_ffn(monkeypatch):
    base_params = {
        "layers": {
            "mlp_1": {
                "gating_einsum": np.arange(6, dtype=np.float32).reshape(2, 3),
                "linear": np.arange(12, dtype=np.float32).reshape(3, 4),
            },
            "final_norm": {"scale": np.array([7.0], dtype=np.float32)},
        }
    }
    model_params = {
        "layers": {
            "mlp_1": {
                "gating_einsum": np.full((2, 3), -2.0, dtype=np.float32),
                "linear": np.full((3, 4), -3.0, dtype=np.float32),
                "expert_0": {
                    "gating_einsum": np.zeros((2, 3), dtype=np.float32),
                    "linear": np.zeros((3, 4), dtype=np.float32),
                },
                "expert_1": {
                    "gating_einsum": np.zeros((2, 3), dtype=np.float32),
                    "linear": np.zeros((3, 4), dtype=np.float32),
                },
                "router": {"kernel": np.full((3, 2), -1.0, dtype=np.float32)},
            },
            "final_norm": {"scale": np.zeros((1,), dtype=np.float32)},
        },
        "state_proj": {"kernel": np.full((5, 6), 9.0, dtype=np.float32)},
    }

    monkeypatch.setattr(moe_weight_loader.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(moe_weight_loader._model, "restore_params", lambda path, restore_type=None: base_params)  # noqa: SLF001

    loader = moe_weight_loader.MoEWeightLoader(
        base_loader=weight_loaders.CheckpointWeightLoader("unused"),
        num_experts=2,
    )
    loaded = loader.load(model_params)

    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["gating_einsum"],
        base_params["layers"]["mlp_1"]["gating_einsum"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["linear"],
        base_params["layers"]["mlp_1"]["linear"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["expert_0"]["gating_einsum"],
        base_params["layers"]["mlp_1"]["gating_einsum"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["expert_1"]["linear"],
        base_params["layers"]["mlp_1"]["linear"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["final_norm"]["scale"],
        base_params["layers"]["final_norm"]["scale"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["router"]["kernel"],
        model_params["layers"]["mlp_1"]["router"]["kernel"],
    )
    np.testing.assert_array_equal(
        loaded["state_proj"]["kernel"],
        model_params["state_proj"]["kernel"],
    )


def test_residual_moe_weight_loader_merges_dense_checkpoint(monkeypatch):
    base_params = {
        "layers": {
            "mlp_1": {
                "gating_einsum": np.arange(6, dtype=np.float32).reshape(2, 3),
                "linear": np.arange(12, dtype=np.float32).reshape(3, 4),
            },
            "final_norm": {"scale": np.array([7.0], dtype=np.float32)},
        }
    }
    model_params = {
        "layers": {
            "mlp_1": {
                "gating_einsum": np.zeros((2, 3), dtype=np.float32),
                "linear": np.zeros((3, 4), dtype=np.float32),
                "expert_0": {
                    "gating_einsum": np.full((2, 3), 5.0, dtype=np.float32),
                    "linear": np.full((3, 4), 6.0, dtype=np.float32),
                },
                "router": {"kernel": np.full((3, 1), -1.0, dtype=np.float32)},
            },
            "final_norm": {"scale": np.zeros((1,), dtype=np.float32)},
        },
    }

    monkeypatch.setattr(moe_weight_loader.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(moe_weight_loader._model, "restore_params", lambda path, restore_type=None: base_params)  # noqa: SLF001

    loader = moe_weight_loader.ResidualMoEWeightLoader(
        base_loader=weight_loaders.CheckpointWeightLoader("unused"),
    )
    loaded = loader.load(model_params)

    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["gating_einsum"],
        base_params["layers"]["mlp_1"]["gating_einsum"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["linear"],
        base_params["layers"]["mlp_1"]["linear"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["final_norm"]["scale"],
        base_params["layers"]["final_norm"]["scale"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["router"]["kernel"],
        model_params["layers"]["mlp_1"]["router"]["kernel"],
    )
    np.testing.assert_array_equal(
        loaded["layers"]["mlp_1"]["expert_0"]["linear"],
        model_params["layers"]["mlp_1"]["expert_0"]["linear"],
    )


def _get_frozen_state(config: pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_moe_freeze_filter_includes_router_trainables():
    config = pi0_moe_config.Pi0MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_config=moe.MoEConfig(num_experts=2, top_k=1, router_z_loss_coeff=1e-3),
    )
    state = _get_frozen_state(config)
    flat_paths = {".".join(path) for path in state}
    assert any("llm" in path for path in flat_paths)
    assert all("router" not in path for path in flat_paths)


def test_pi0_residual_moe_freeze_filter_includes_router_and_expert_trainables():
    config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_config=moe.ResidualMoEConfig(num_experts=2, expert_hidden_dim=16),
    )
    state = _get_frozen_state(config)
    flat_paths = {".".join(path) for path in state}
    assert any("llm" in path for path in flat_paths)
    assert all("router" not in path for path in flat_paths)
    assert all("expert_0" not in path for path in flat_paths)
    assert all("expert_1" not in path for path in flat_paths)


def test_top1_load_balance_loss_penalizes_collapsed_routing():
    balanced_logits = jnp.array([[[8.0, -8.0], [-8.0, 8.0]]], dtype=jnp.float32)
    collapsed_logits = jnp.array([[[8.0, -8.0], [8.0, -8.0]]], dtype=jnp.float32)

    balanced_loss = moe._load_balance_loss(balanced_logits)  # noqa: SLF001
    collapsed_loss = moe._load_balance_loss(collapsed_logits)  # noqa: SLF001

    assert balanced_loss < collapsed_loss


def test_residual_scale_zero_matches_dense_pi0_sampling(monkeypatch):
    key = jax.random.key(0)
    dense_config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    residual_config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_config=moe.ResidualMoEConfig(num_experts=2, expert_hidden_dim=16, residual_scale=0.0),
    )

    dense_model = dense_config.create(key)
    dense_params = nnx.state(dense_model).to_pure_dict()
    residual_ref_params = nnx.state(residual_config.create(key)).to_pure_dict()

    monkeypatch.setattr(moe_weight_loader.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(moe_weight_loader._model, "restore_params", lambda path, restore_type=None: dense_params)  # noqa: SLF001
    loader = moe_weight_loader.ResidualMoEWeightLoader(
        base_loader=weight_loaders.CheckpointWeightLoader("unused"),
    )
    residual_params = loader.load(residual_ref_params)
    residual_model = residual_config.load(residual_params)

    batch_size = 2
    obs = dense_config.fake_obs(batch_size)
    noise = jnp.ones((batch_size, dense_config.action_horizon, dense_config.action_dim), dtype=jnp.float32)

    dense_actions = nnx_utils.module_jit(dense_model.sample_actions)(key, obs, num_steps=2, noise=noise)
    residual_actions = nnx_utils.module_jit(residual_model.sample_actions)(key, obs, num_steps=2, noise=noise)

    np.testing.assert_allclose(np.asarray(residual_actions), np.asarray(dense_actions), atol=1e-6)


def test_pi0_moe_no_moe_layers_matches_dense_pi0_sampling(monkeypatch):
    key = jax.random.key(0)
    dense_config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    moe_config = pi0_moe_config.Pi0MoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[],
        moe_config=moe.MoEConfig(num_experts=2, top_k=1, router_z_loss_coeff=1e-3),
    )

    dense_model = dense_config.create(key)
    dense_params = nnx.state(dense_model).to_pure_dict()
    moe_ref_params = nnx.state(moe_config.create(key)).to_pure_dict()

    monkeypatch.setattr(moe_weight_loader.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(moe_weight_loader._model, "restore_params", lambda path, restore_type=None: dense_params)  # noqa: SLF001
    loader = moe_weight_loader.MoEWeightLoader(
        base_loader=weight_loaders.CheckpointWeightLoader("unused"),
        num_experts=2,
    )
    moe_params = loader.load(moe_ref_params)
    moe_model = moe_config.load(moe_params)

    batch_size = 2
    obs = dense_config.fake_obs(batch_size)
    noise = jnp.ones((batch_size, dense_config.action_horizon, dense_config.action_dim), dtype=jnp.float32)

    dense_actions = nnx_utils.module_jit(dense_model.sample_actions)(key, obs, num_steps=2, noise=noise)
    moe_actions = nnx_utils.module_jit(moe_model.sample_actions)(key, obs, num_steps=2, noise=noise)

    np.testing.assert_allclose(np.asarray(moe_actions), np.asarray(dense_actions), atol=1e-6)


def test_no_moe_layers_matches_dense_pi0_sampling(monkeypatch):
    key = jax.random.key(0)
    dense_config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    residual_config = pi0_residual_moe_config.Pi0ResidualMoEConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        moe_layers=[],
        moe_config=moe.ResidualMoEConfig(num_experts=2, expert_hidden_dim=16, residual_scale=1.0),
    )

    dense_model = dense_config.create(key)
    dense_params = nnx.state(dense_model).to_pure_dict()
    residual_ref_params = nnx.state(residual_config.create(key)).to_pure_dict()

    monkeypatch.setattr(moe_weight_loader.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(moe_weight_loader._model, "restore_params", lambda path, restore_type=None: dense_params)  # noqa: SLF001
    loader = moe_weight_loader.ResidualMoEWeightLoader(
        base_loader=weight_loaders.CheckpointWeightLoader("unused"),
    )
    residual_params = loader.load(residual_ref_params)
    residual_model = residual_config.load(residual_params)

    batch_size = 2
    obs = dense_config.fake_obs(batch_size)
    noise = jnp.ones((batch_size, dense_config.action_horizon, dense_config.action_dim), dtype=jnp.float32)

    dense_actions = nnx_utils.module_jit(dense_model.sample_actions)(key, obs, num_steps=2, noise=noise)
    residual_actions = nnx_utils.module_jit(residual_model.sample_actions)(key, obs, num_steps=2, noise=noise)

    np.testing.assert_allclose(np.asarray(residual_actions), np.asarray(dense_actions), atol=1e-6)
