import dataclasses
import functools
import logging
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

try:
    from scripts import train as base_train
except ModuleNotFoundError:
    import train as base_train


def _flatten_moe_metrics(moe_metrics: dict[str, at.Array]) -> dict[str, at.Array]:
    expert_usage = moe_metrics["expert_usage"]
    info = {
        "moe/router_prob_mean": jnp.mean(moe_metrics["router_prob_mean"]),
        "moe/router_prob_variance": jnp.mean(moe_metrics["router_prob_variance"]),
        "moe/router_entropy": jnp.mean(moe_metrics["router_entropy"]),
        "moe/router_logits_mean": jnp.mean(moe_metrics["router_logits_mean"]),
        "moe/router_logits_std": jnp.mean(moe_metrics["router_logits_std"]),
        "moe/router_logits_max": jnp.max(moe_metrics["router_logits_max"]),
        "moe/router_z_loss": jnp.mean(moe_metrics["router_z_loss"]),
        "moe/load_balance_loss": jnp.mean(moe_metrics["load_balance_loss"]),
    }

    global_usage = jnp.mean(expert_usage, axis=0)
    info["moe/global_expert_usage"] = global_usage
    for expert_idx in range(expert_usage.shape[1]):
        info[f"moe/global_expert_{expert_idx}_usage"] = global_usage[expert_idx]

    for layer_idx in range(expert_usage.shape[0]):
        info[f"moe/layer_{layer_idx}/expert_usage"] = expert_usage[layer_idx]
        info[f"moe/layer_{layer_idx}/router_entropy"] = moe_metrics["router_entropy"][layer_idx]
        info[f"moe/layer_{layer_idx}/router_prob_variance"] = moe_metrics["router_prob_variance"][layer_idx]
        info[f"moe/layer_{layer_idx}/load_balance_loss"] = moe_metrics["load_balance_loss"][layer_idx]
        for expert_idx in range(expert_usage.shape[1]):
            info[f"moe/layer_{layer_idx}/expert_{expert_idx}_usage"] = expert_usage[layer_idx, expert_idx]

    return info


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss, moe_metrics = model.compute_loss_with_moe_metrics(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), moe_metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, moe_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(_flatten_moe_metrics(moe_metrics))
    return new_state, info


def main(config: _config.TrainConfig):
    base_train.init_logging()
    logging.info(f"Running on: {platform.node()}")

    if not hasattr(config.model, "moe_config"):
        raise ValueError("scripts/train_moe.py requires a MoE model config with a moe_config field.")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    base_train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = base_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items() if np.ndim(v) == 0)
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
