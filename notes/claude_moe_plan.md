# Pi0-MoE: Mixture-of-Experts Action Expert

## Context
Replace the single feed-forward network in pi0.5's action expert with a Mixture-of-Experts (MoE)
FFW layer. Each MoE expert is initialized from the pre-trained action expert weights. No existing
openpi files are modified — everything is implemented as new files in the models/ directory,
with a minimal addition to config.py's `_CONFIGS` list.

---

## Architecture Background (from codebase exploration)

### Current dual-expert structure
`gemma.Module` takes a list of `Config` objects (one per "token-type expert"):
- Expert 0 (PaliGemma backbone, `gemma_2b`): processes image + language tokens
- Expert 1 (Action Expert, `gemma_300m_lora`): processes action + timestep tokens

Each `Block` in `gemma.py` creates per-expert sub-modules named with the `_name(name, i)` pattern:
- `mlp`   → expert 0 feed-forward (PaliGemma FFW)
- `mlp_1` → expert 1 feed-forward (Action Expert FFW)

The **Action Expert FFW** (`lora.FeedForward`) has these parameters per layer:
```
PaliGemma/llm/layers/{i}/mlp_1/gating_einsum        (2, 1024, 4096)
PaliGemma/llm/layers/{i}/mlp_1/linear               (4096, 1024)
PaliGemma/llm/layers/{i}/mlp_1/gating_einsum_lora_a (2, 1024, 32)  # if LoRA
PaliGemma/llm/layers/{i}/mlp_1/gating_einsum_lora_b (2, 32, 4096)
PaliGemma/llm/layers/{i}/mlp_1/linear_lora_a        (4096, 32)
PaliGemma/llm/layers/{i}/mlp_1/linear_lora_b        (32, 1024)
```

The **MoE replacement** will introduce `N` expert sub-modules + a router under `mlp_1/`:
```
PaliGemma/llm/layers/{i}/mlp_1/router/kernel        (1024, N)
PaliGemma/llm/layers/{i}/mlp_1/expert_0/gating_einsum  (2, 1024, 4096)
PaliGemma/llm/layers/{i}/mlp_1/expert_0/linear         (4096, 1024)
...
PaliGemma/llm/layers/{i}/mlp_1/expert_{N-1}/...
```

---

## New Files

### 1. `src/openpi/models/moe.py`
Pure flax.linen modules — no dependency on or modification of existing files.

**`MoEConfig` dataclass**
```python
@dataclasses.dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2               # sparse routing: activate top_k of num_experts
    router_z_loss_coeff: float = 1e-3   # auxiliary load-balancing loss coefficient
```

**`MoEFeedForward(nn.Module)`**
Drop-in replacement for `lora.FeedForward`. Params layout:
- `router/kernel`: `(features, num_experts)`  — linear router, initialized to zeros
- `expert_{k}/*`: each is a standard `lora.FeedForward` (same field names as current)

Forward pass:
1. Router logits: `x @ router/kernel`  → `(batch, seq, num_experts)`
2. Top-K selection + softmax renormalization over top-K weights
3. Weighted sum of the top-K experts' outputs
4. Returns `(combined_output, router_logits)` — caller uses router_logits for z-loss

**`MoEBlock(nn.Module)`**
Mirrors `gemma.Block` exactly, except for expert index 1's FFW step:
- All expert indices `i != 1`: use `lora.FeedForward` (unchanged)
- Expert index `i == 1`: use `MoEFeedForward`
- Takes extra `moe_config: MoEConfig` field

**`MoEModule(nn.Module)`**
Mirrors `gemma.Module` exactly, using `MoEBlock` instead of `Block`.
Takes extra `moe_config: MoEConfig` field passed through to each block.
Returns `(outputs, all_router_logits)` from `__call__` so z-loss can be computed.

---

### 2. `src/openpi/models/pi0_moe.py`
Mirrors `pi0.py` with one change: instantiates `moe.MoEModule` instead of `_gemma.Module`.

```python
class Pi0MoE(_model.BaseModel):
    def __init__(self, config: "Pi0MoEConfig", rngs: nnx.Rngs): ...

    def compute_loss(self, rng, obs, actions, train=False):
        # Standard pi0.5 flow matching loss
        # + moe_config.router_z_loss_coeff * router_z_loss(all_router_logits)
        ...
```

Router z-loss: `mean(log(sum(exp(logits)))^2)` — standard Mesh-TensorFlow z-loss.

---

### 3. `src/openpi/models/pi0_moe_config.py`
```python
@dataclasses.dataclass(frozen=True)
class Pi0MoEConfig(_model.BaseModelConfig):
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"
    moe_config: moe.MoEConfig = dataclasses.field(default_factory=moe.MoEConfig)
    pi05: bool = True
    dtype: str = "bfloat16"
    # ... same action_dim, action_horizon, max_token_len as Pi0Config

    def create(self, rng) -> Pi0MoE: ...
    def get_freeze_filter(self): ...  # same logic as Pi0Config
```

---

### 4. `src/openpi/training/moe_weight_loader.py`
Custom `WeightLoader` that:
1. Loads the base pi0.5 checkpoint via `CheckpointWeightLoader`
2. Fans out each action expert FFW layer to `N` expert sub-dicts:
   ```
   mlp_1/{gating_einsum,linear,...}  →  mlp_1/expert_0/{...}, ..., mlp_1/expert_{N-1}/{...}
   ```
3. Inserts zero-initialized router weights at `mlp_1/router/kernel`

Uses `flax.traverse_util.flatten_dict` / `unflatten_dict` (same pattern as `_load_weights_and_validate` in `sft_train.py`).

---

### 5. Minimal addition to `src/openpi/training/config.py`
Add imports at top of `_CONFIGS` section and one entry to the list:
```python
from openpi.models.pi0_moe_config import Pi0MoEConfig
from openpi.models.moe import MoEConfig
from openpi.training.moe_weight_loader import MoEWeightLoader

# In _CONFIGS list:
TrainConfig(
    name="pi05_libero_moe",
    model=Pi0MoEConfig(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        moe_config=MoEConfig(num_experts=4, top_k=2, router_z_loss_coeff=1e-3),
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    weight_loader=MoEWeightLoader(
        base_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
    freeze_filter=nnx.Any(
        nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
        nnx.All(nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                nnx.Not(nnx_utils.PathRegex(".*router.*"))),
    ),
    ema_decay=None,
)
```

CLI overrides work naturally via tyro, e.g.:
```bash
bash shells/run_sft.sh --config_name pi05_libero_moe \
    --model.moe_config.num_experts 8 \
    --model.moe_config.top_k 2
```

---

## Weight Initialization Detail

For each of the 18 transformer layers, `MoEWeightLoader`:
1. Reads `mlp_1/gating_einsum` and `mlp_1/linear` (and LoRA params if present) from base ckpt
2. Copies them verbatim to `mlp_1/expert_0/`, `mlp_1/expert_1/`, ... `mlp_1/expert_{N-1}/`
3. Drops the top-level `mlp_1/gating_einsum` and `mlp_1/linear` keys
4. Inserts `mlp_1/router/kernel = zeros((1024, N))`

All N experts start identical to the original action expert. The router starts uniform (all zero logits → equal probability), so initial behavior is close to the original single-FFW model (weighted average of identical experts = same as one expert).

---

## Training: freeze filter & LoRA

`Pi0MoEConfig.get_freeze_filter()` logic:
- PaliGemma backbone: fully frozen (`.*llm.*` excluding `.*llm.*_1.*`)
- Action expert base weights: frozen (`.*llm.*_1.*` excluding `.*lora.*` and `.*router.*`)
- Action expert LoRA adapters (`.*lora.*`): **trainable**
- Router kernel (`.*router.*`): **always trainable**

Concretely:
```python
freeze_filter = nnx.Any(
    # Freeze all PaliGemma LLM params
    nnx.All(PathRegex(".*llm.*"), nnx.Not(PathRegex(".*llm.*_1.*"))),
    # Freeze action expert base weights (not LoRA, not router)
    nnx.All(PathRegex(".*llm.*_1.*"),
            nnx.Not(PathRegex(".*lora.*")),
            nnx.Not(PathRegex(".*router.*"))),
)
```

**LoRA on each expert**: `MoEFeedForward` passes `lora_config` to each `lora.FeedForward(name=f"expert_{k}")`. The lora_config comes from `action_expert_config.lora_configs.get("ffn")` (rank=32 from `gemma_300m_lora`). Each expert's LoRA params are thus named:
```
mlp_1/expert_k/gating_einsum_lora_a   (2, 1024, 32)
mlp_1/expert_k/gating_einsum_lora_b   (2, 32, 4096)
mlp_1/expert_k/linear_lora_a          (4096, 32)
mlp_1/expert_k/linear_lora_b          (32, 1024)
```
The pattern `.*lora.*` correctly matches these paths, keeping them trainable.

---

## Files to Create / Modify

| File | Change |
|---|---|
| `src/openpi/models/moe.py` | **New** — MoEConfig, MoEFeedForward, MoEBlock, MoEModule |
| `src/openpi/models/pi0_moe.py` | **New** — Pi0MoE model class |
| `src/openpi/models/pi0_moe_config.py` | **New** — Pi0MoEConfig dataclass |
| `src/openpi/training/moe_weight_loader.py` | **New** — MoEWeightLoader |
| `src/openpi/training/config.py` | **Minimal add** — append `pi05_libero_moe` entry to `_CONFIGS` |

No existing model files (gemma.py, pi0.py, lora.py, pi0_config.py, model.py) are touched.

---

## Confirmed design decisions
- **num_experts / top_k**: Configurable via CLI (tyro exposes `Pi0MoEConfig.moe_config.num_experts` etc.)
- **Default**: `num_experts=4, top_k=2`
- **Router z-loss**: Yes — `z_loss_coeff=1e-3` added to flow-matching loss
- **LoRA on each expert**: Yes — each `expert_k` sub-FFW uses `lora_config=action_expert_config.lora_configs.get("ffn")`, inheriting rank-32 LoRA from `gemma_300m_lora`

---

## Verification
1. `python -c "from openpi.models.pi0_moe_config import Pi0MoEConfig; c = Pi0MoEConfig(); m = c.create(jax.random.key(0))"` — model instantiates
2. `uv run scripts/sft_train.py --config_name pi05_libero_moe --list_tasks` — config resolves
3. A short 10-step training run completes without error
4. Checkpoint param tree contains `mlp_1/router/kernel` and `mlp_1/expert_0/gating_einsum` etc.
5. `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero_moe ...` serves successfully

