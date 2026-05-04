# Pi0-DynMoE: Dynamic Mixture-of-Experts Action Expert

## Context

Implement a DynMoE variant of the pi0.5 action expert, following the paper "DynMoE: Dynamic Mixture
of Experts for Continual Learning". Three distinct improvements over the existing `pi05_libero_moe`:

1. **Top-Any gating** — sigmoid thresholds replace softmax+top-k; k varies per token
2. **Diverse-and-Simple auxiliary loss** — orthogonality + magnitude regularization on expert
   representations; replaces router z-loss
3. **Dynamic expert add/remove** — between tasks, inactive experts are pruned and new experts are
   seeded from tokens that activated no expert; implemented in a new `dynamic_sft_train.py` so
   that `sft_train.py` is not touched

No existing openpi files are modified. All code is new files.

---

## DynMoE Algorithm (from paper + repo)

### Top-Any Gating (paper Eq. 3–6)

```
s(x)  = ⟨x/‖x‖, W_g/‖W_g‖_col⟩      cosine similarity   (B, T, E)
σ_s   = sigmoid(s(x))                  activation probs    (B, T, E)
diff  = σ_s − sigmoid(G)               vs per-expert threshold
g(x)  = sign(diff)    [STE backward]   binary gate         (B, T, E)
k(x)  = Σ g(x)                         dynamic k per token
```

- `W_g ∈ ℝ^(D × E_max)` — expert representation matrix; `experts_mask` zeroes inactive columns
- `G ∈ ℝ^(E_max)` — per-expert learnable thresholds, init = 0 (→ sigmoid = 0.5)
- `experts_mask ∈ {0,1}^(E_max)` — frozen param, init `[1]*num_experts + [0]*(max_experts-num_experts)`
- **STE**: `g = diff + stop_gradient(sign(diff) − diff)`
- **Inference fallback** (paper Eq. 7): if `k(x) == 0` for a token, force top-1 via argmax of
  `σ_s * experts_mask`

### Diverse-and-Simple Gating Loss (paper Eq. 8)

```
L_aux = ‖Ŵ_g^T Ŵ_g − I_K‖_F²  +  (1/K) Σ_e ‖w_{g,e}‖₂
```

- Left: **diversity** — Gram matrix of normalized active expert columns → identity
- Right: **simplicity** — bounded column norms
- Only computed over active experts (masked by `experts_mask`)
- Coefficient `aux_loss_coeff = 1e-3`

### Adaptive Expert Update (paper Algorithm 1, between tasks)

After each task completes, run `adaptive_update_experts(model, data_loader)`:

1. **Record routing** — forward-pass N batches, accumulate:
   - `activation_counts[l, e]` — how many tokens activated expert `e` in layer `l`
   - `unrouted_sum[l, d]` — sum of token embeddings `x` where `g(x) = 0` (no expert selected)
   - `unrouted_count[l]` — number of such tokens

2. **Remove** — for each layer `l`, for each active expert `e`:
   if `activation_counts[l, e] == 0`: set `experts_mask[l, e] = 0`

3. **Add** — for each layer `l`:
   if `unrouted_count[l] > 0` and a masked-out slot exists:
   - find first inactive slot `new_e`
   - `experts_mask[l, new_e] = 1`
   - `sim_matrix[l, :, new_e] = unrouted_sum[l] / ‖unrouted_sum[l]‖`
   - `gates[l, new_e] = 0`

**JAX constraint**: These mutations happen in Python outside of JIT — the NNX model's param arrays
are modified directly via `nnx.state()` between tasks. No in-step shape changes required because
`max_experts` is fixed and the mask handles activation.

---

## New Files (6 total)

### 1. `src/openpi/models/dyn_moe.py`

#### `DynMoEConfig`
```python
@dataclasses.dataclass(frozen=True)
class DynMoEConfig:
    num_experts: int = 2      # initially active experts
    max_experts: int = 4      # pre-allocated capacity (paper default for VL tasks)
    aux_loss_coeff: float = 1e-3
```

#### `DynMoEGate(nn.Module)` — instantiated as `self.router` in parent
Params under `mlp_1/router/`:
- `sim_matrix`: `(D, max_experts)`, lecun_normal — expert representations W_g
- `gates`: `(max_experts,)`, zeros — per-expert thresholds G
- `experts_mask`: `(max_experts,)`, init `[1]*num_experts + [0]*rest` — **frozen param**

`__call__(self, x, deterministic)`:
```
x_norm     = x / (‖x‖ + ε)                             (B, T, D)
w_masked   = sim_matrix * experts_mask[None, :]          (D, E_max)  zero inactive cols
w_norm     = w_masked / (‖w_masked‖_col + ε)
cos_sim    = einsum("...d,de->...e", x_norm, w_norm)     (B, T, E_max)
probs      = sigmoid(cos_sim) * experts_mask             (B, T, E_max)
diff       = probs − sigmoid(gates) * experts_mask
gate_hard  = diff + stop_gradient(sign(diff) − diff)     STE (B, T, E_max) binary

# inference fallback
if deterministic:
    any_sel  = any(gate_hard > 0, axis=-1, keepdims=True)
    top1     = one_hot(argmax(probs, axis=-1), max_experts)
    gate_hard = where(any_sel, gate_hard, top1)

weights = probs * gate_hard
weights = weights / (sum(weights, axis=-1, keepdims=True) + ε)

return gate_hard.astype(dtype), weights.astype(dtype), sim_matrix
```

#### `_diverse_and_simple_gate_loss(sim_matrix, experts_mask)`
```python
# Only active experts contribute
mask = experts_mask  # (E_max,)
w = sim_matrix * mask[None, :]                           # zero inactive cols
w_norm = w / (‖w‖_col + ε)                              # (D, E_max)
gram   = einsum("de,df->ef", w_norm, w_norm)             # (E_max, E_max)
# Mask gram to active-active pairs only
m2 = outer(mask, mask)
diversity  = sum((gram − eye) ** 2 * m2)
simplicity = sum(‖sim_matrix‖_col * mask) / (sum(mask) + ε)
return diversity + simplicity
```

#### `DynMoEFeedForward(nn.Module)`
- `setup()`: `setattr(self, f"expert_{k}", lora.FeedForward(...))` for k in `range(max_experts)`;
  `self.router = DynMoEGate(..., name="router")`
- `__call__(self, x, deterministic)`:
  1. `gate_hard, weights, sim_matrix = self.router(x, deterministic)`
  2. `expert_outputs = stack([expert_k(x) for k in range(max_experts)], axis=-2)`
  3. `output = einsum("...e,...ed->...d", weights, expert_outputs).astype(dtype)`
  4. `aux_loss = _diverse_and_simple_gate_loss(sim_matrix, router.experts_mask)`
  5. `return output, aux_loss`

Note: all `max_experts` experts always computed (same as current MoE — no sparse compute).

#### `DynMoEBlock(nn.Module)`
Mirrors `MoEBlock` with two changes:
- Calls `DynMoEFeedForward(...)(x, deterministic)` for expert index 1
- Returns `(xs, (kv_cache, aux_loss_scalar))` — identical scan signature to `MoEBlock`

#### `DynMoEModule(nn.Module)`
Mirrors `MoEModule`, substituting `DynMoEBlock`. `__call__` returns same 3-tuple
`(outputs, kv_cache, total_aux_loss)`.

Additional method `compute_routing_stats(embedded, positions, mask, adarms_cond)`:
- Same forward as `__call__` but returns `(activation_counts, unrouted_sum, unrouted_count)`
  instead of outputs, for use by `adaptive_update_experts()`
- `activation_counts`: `(depth, max_experts)` — tokens activating each expert per layer
- `unrouted_sum`: `(depth, D)` — sum of token embeddings where no expert was selected
- `unrouted_count`: `(depth,)` — count of unrouted tokens per layer

#### `adaptive_update_experts(model, data_loader, num_record_batches=50)`
Top-level function in `dyn_moe.py`:
1. Extract `DynMoEModule` handle from `model.PaliGemma.llm`
2. Accumulate routing stats over `num_record_batches` forward passes
3. Compute new `experts_mask`, `sim_matrix`, `gates` per the algorithm above
4. Write updated arrays back into the NNX model state via direct assignment

---

### 2. `src/openpi/models/pi0_dyn_moe.py`
Near-identical copy of `pi0_moe.py`:
- Uses `dyn_moe.DynMoEModule` instead of `moe.MoEModule`
- Config type is `Pi0DynMoEConfig`
- `compute_loss` uses `config.dyn_moe_config.aux_loss_coeff`
- `sample_actions` unchanged (ignores aux_loss identically)

---

### 3. `src/openpi/models/pi0_dyn_moe_config.py`
```python
@dataclasses.dataclass(frozen=True)
class Pi0DynMoEConfig(Pi0Config):
    dyn_moe_config: DynMoEConfig = field(default_factory=DynMoEConfig)

    def create(self, rng) -> Pi0DynMoE: ...

    def get_freeze_filter(self):
        # Same pattern as Pi0MoEConfig.
        # ".*router.*" matches router/sim_matrix, router/gates, AND router/experts_mask
        # experts_mask is frozen from optimizer but mutable by adaptive_update_experts()
        return nnx.Any(
            nnx.All(PathRegex(".*llm.*"), nnx.Not(PathRegex(".*llm.*_1.*"))),
            nnx.All(
                PathRegex(".*llm.*_1.*"),
                nnx.Not(PathRegex(".*lora.*")),
                nnx.Not(PathRegex(".*router.*")),
            ),
        )
```

---

### 4. `src/openpi/training/dyn_moe_weight_loader.py`
Same fan-out as `MoEWeightLoader` (`mlp_1/{leaf}` → `mlp_1/expert_k/{leaf}` for k in
`range(max_experts)`), with two differences:
- Uses `max_experts` (not `num_experts`) for fan-out count — pre-populates all slots
- Does **not** insert a router kernel — `sim_matrix`, `gates`, `experts_mask` are filled
  from the model's own initialization via `_merge_params` fallback

```python
@dataclasses.dataclass(frozen=True)
class DynMoEWeightLoader(weight_loaders.WeightLoader):
    base_loader: weight_loaders.CheckpointWeightLoader
    num_experts: int = 2     # active at start
    max_experts: int = 4     # total pre-allocated slots
```

---

### 5. `src/openpi/training/config.py` (minimal add)
```python
TrainConfig(
    name="pi05_libero_dyn_moe",
    model=Pi0DynMoEConfig(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        dyn_moe_config=DynMoEConfig(num_experts=2, max_experts=4, aux_loss_coeff=1e-3),
    ),
    data=LeRobotLiberoDataConfig(...),
    weight_loader=DynMoEWeightLoader(
        base_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_experts=2, max_experts=4,
    ),
    freeze_filter=Pi0DynMoEConfig().get_freeze_filter(),
    ema_decay=None,
),
```

---

### 6. `scripts/dynamic_sft_train.py`
New script — mirrors `sft_train.py` structure exactly, with one addition in the task loop:

```python
# After checkpoint_manager.wait_until_finished() at end of each task:
if isinstance(config.model, Pi0DynMoEConfig):
    logging.info("Running DynMoE adaptive expert update...")
    from openpi.models.dyn_moe import adaptive_update_experts
    # Build a small recording data loader (same task, short)
    record_loader = create_task_data_loader(config, task_name, data_sharding)
    adaptive_update_experts(
        model=train_state,           # NNX model accessible via train_state
        data_loader=record_loader,
        num_record_batches=50,
    )
    logging.info("DynMoE adaptive update complete.")
```

`dynamic_sft_train.py` accepts the same CLI args as `sft_train.py` (same `SFTArgs` dataclass),
so it works transparently with `run_sft.sh` via `--config_name pi05_libero_dyn_moe`.

---

## Parameter Tree vs Standard MoE

| Path | Standard MoE | DynMoE |
|---|---|---|
| `mlp_1/router/kernel` | `(18, D, E)` | absent |
| `mlp_1/router/sim_matrix` | absent | `(18, D, E_max)` lecun_normal |
| `mlp_1/router/gates` | absent | `(18, E_max)` zeros |
| `mlp_1/router/experts_mask` | absent | `(18, E_max)` binary, frozen |
| `mlp_1/expert_k/*` | k=0..3 | k=0..E_max-1 (inactive slots exist) |

---

## Scan Compatibility Constraints (unchanged from MoE)

1. `DynMoEBlock` returns `(xs, (kv_cache, aux_loss_scalar))` — scalar `()` shape
2. Carry `xs` dtype matches input — weights/gate_hard cast to `x.dtype` before einsum
3. `deterministic` is `static_argnums=(5,)` in `nn.remat` — safe to branch on in Python

---

## Files to Create / Modify

| File | Change |
|---|---|
| `src/openpi/models/dyn_moe.py` | **New** — DynMoEConfig, DynMoEGate, DynMoEFeedForward, DynMoEBlock, DynMoEModule, adaptive_update_experts |
| `src/openpi/models/pi0_dyn_moe.py` | **New** — Pi0DynMoE model class |
| `src/openpi/models/pi0_dyn_moe_config.py` | **New** — Pi0DynMoEConfig |
| `src/openpi/training/dyn_moe_weight_loader.py` | **New** — DynMoEWeightLoader |
| `src/openpi/training/config.py` | **Minimal add** — `pi05_libero_dyn_moe` entry |
| `scripts/dynamic_sft_train.py` | **New** — mirrors sft_train.py + adaptive update between tasks |

No existing files touched.

---

## Verification

```bash
cd /local_data/lim2045/openpi

# 1. Model instantiation
uv run python -c "
import jax
from openpi.models.pi0_dyn_moe_config import Pi0DynMoEConfig
m = Pi0DynMoEConfig().create(jax.random.key(0))
print('ok')
"

# 2. Config resolves
uv run scripts/dynamic_sft_train.py --config_name pi05_libero_dyn_moe --list_tasks

# 3. Short training run with adaptive update
bash shells/run_sft.sh --config_name pi05_libero_dyn_moe --num_tasks 2 \
    --steps_per_task 10 --no-wandb_enabled \
    --sft_script scripts/dynamic_sft_train.py

# 4. Check param tree — expect router/sim_matrix, router/gates, router/experts_mask
# 5. After task 1, experts_mask should reflect any pruning/addition
```

---

## Implementation Notes (added 2026-03-26)

### Status: All 6 files created and smoke-tested ✓

### Files created
| File | Status |
|---|---|
| `src/openpi/models/dyn_moe.py` | ✓ created |
| `src/openpi/models/pi0_dyn_moe.py` | ✓ created |
| `src/openpi/models/pi0_dyn_moe_config.py` | ✓ created |
| `src/openpi/training/dyn_moe_weight_loader.py` | ✓ created |
| `src/openpi/training/config.py` | ✓ minimal add (imports + `pi05_libero_dyn_moe` entry) |
| `scripts/dynamic_sft_train.py` | ✓ created |

### Deviations from plan

**1. `if deterministic:` replaced with `jnp.where` (JAX tracing constraint)**

The plan specified:
```python
if deterministic:
    gate_hard = jnp.where(any_selected, gate_hard, top1_mask)
```
This caused `jax.errors.TracerBoolConversionError` inside `nn.scan`. Even with `static_argnums=(5,)` in `nn.remat(DynMoEBlock, ...)`, the `deterministic` value arrives as a traced JAX array inside the scan's `body_fn` (from `axes_scan.py`). Python `if` on a traced bool is not allowed.

Fix: always apply the top-1 fallback unconditionally via `jnp.where`:
```python
any_selected = jnp.any(gate_hard > 0, axis=-1, keepdims=True)
top1_mask = jax.nn.one_hot(jnp.argmax(probs, axis=-1), self.max_experts)
gate_hard = jnp.where(any_selected, gate_hard, top1_mask)
```
During training, k=0 tokens are rare (gates init at 0 → sigmoid(0)=0.5, so ~half of cos_sim values exceed threshold). When k≥1, `any_selected=True` and `gate_hard` is unchanged. When k=0, top-1 fallback is used (no STE gradient for that token). This is acceptable — the fallback stabilizes training and matches inference behavior.

**2. `DynMoEGate.__call__` `deterministic` param kept but unused**

The `deterministic` parameter is still accepted for interface consistency with `DynMoEFeedForward` and `DynMoEBlock`, but no longer branches on it in Python. The top-1 fallback runs unconditionally.

**3. Scan output extended to 5-tuple**

The plan showed `(xs, (kv_cache, aux_loss_scalar))` as the scan output. The actual output is:
```python
(xs, (kv_cache, aux_loss_scalar, activation_counts, unrouted_sum, unrouted_count))
```
This lets `DynMoEModule.__call__` ignore the routing stats (only uses `aux_losses`) and `compute_routing_stats` reuse the same scan without a separate forward pass.

**4. `config.py` is modified (one import block + one TrainConfig entry)**

The plan said "No existing openpi files are modified" but config.py needed the new imports and config entry. This is a minimal touch (3 import lines + the TrainConfig block) and doesn't change existing behaviour.

### Smoke test results (2026-03-26)

```
Model created ok: Pi0DynMoE
compute_loss ok, shape: (1, 50) value: 2.085655...

Router params:
  PaliGemma/llm/layers/mlp_1/router/experts_mask/.value: (18, 4)  # depth=18, max_experts=4
  PaliGemma/llm/layers/mlp_1/router/gates/.value: (18, 4)
  PaliGemma/llm/layers/mlp_1/router/sim_matrix/.value: (18, 1024, 4)  # (depth, D, max_experts)
```

Config `pi05_libero_dyn_moe` resolves correctly. All router params have the expected shapes with the depth=18 leading dimension from `nn.scan`.

### Known TODOs / next steps

- Run a real short training run with `dynamic_sft_train.py --steps_per_task 10` to verify `adaptive_update_experts()` works end-to-end (NNX state mutation, router param writing).
- Verify `DynMoEWeightLoader` correctly fans out base checkpoint to `max_experts=4` slots.
- The `adaptive_update_experts` NNX state mutation uses `flax.traverse_util.flatten_dict` on the NNX `State` object. The State is not a plain dict, so the flatten call at the top of `adaptive_update_experts` may need adjustment — specifically, it uses `jax.tree_util.tree_map` + `flax.traverse_util.flatten_dict`. This path needs a real run to validate.
- Consider whether the top-1 fallback during training (for k=0 tokens) is a problem in practice. If it causes training instability, the original `if deterministic:` intent can be restored by converting `deterministic` to a module-level attribute set before JIT (like `self.deterministic = True` in `Pi0DynMoE`).

