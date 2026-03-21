# Pi0-MoE Architecture

## Overview

Pi0-MoE is a variant of Pi0.5 (the pi0.5 flow-matching robot policy) in which the
action expert's feed-forward network (FFW) in every transformer layer is replaced
with a **sparse Mixture-of-Experts (MoE) FFW layer**.  Everything else — the
PaliGemma vision-language backbone, the SigLIP image encoder, the adaRMSNorm
timestep conditioning, the flow-matching training objective — is identical to Pi0.5.

The design goal is to give the action expert more representational capacity without
increasing the per-forward-pass compute: only `top_k` out of `num_experts` experts
are activated per token per layer.

---

## System Architecture

### Dual-Expert Transformer (inherited from Pi0.5)

The LLM backbone is a shared transformer that processes two token streams
simultaneously via a "dual expert" design:

| Expert index | Token stream | Config | Weights |
|---|---|---|---|
| 0 | Image + language tokens (prefix) | `gemma_2b` — 2B params | PaliGemma checkpoint, **frozen** |
| 1 | Action + timestep tokens (suffix) | `gemma_300m_lora` — 300M params | pi05_base checkpoint, partially frozen |

Both token streams share the same attention mechanism (cross-stream attention is
possible due to concatenated positions and masks), but each has its own
feed-forward network, normalization layers, and QKV projections.  The transformer
depth is 18 layers for both variants.

### What Pi0-MoE Changes

Only expert 1's FFW is replaced.  In every one of the 18 transformer layers:

```
Before (Pi0.5):
  action tokens → RMSNorm → lora.FeedForward  → residual add

After (Pi0-MoE):
  action tokens → RMSNorm → MoEFeedForward    → residual add
                              ├─ router (linear)
                              ├─ expert_0 (lora.FeedForward)
                              ├─ expert_1 (lora.FeedForward)
                              ├─ expert_2 (lora.FeedForward)
                              └─ expert_3 (lora.FeedForward)
```

Expert 0's FFW (PaliGemma backbone) is not touched.

---

## MoE Feed-Forward Layer (`MoEFeedForward`)

### Forward Pass

Given input `x` of shape `(B, T, D)` where `D = 1024` (action expert width):

1. **Router**: `logits = x @ W_router`  →  shape `(B, T, num_experts)`
   - `W_router` is `(D, num_experts)` = `(1024, 4)`, zero-initialized
   - Output is float32 for numerical stability (input may be bfloat16)

2. **Top-K selection**: select the `top_k = 2` experts with the highest logit per token
   - `top_k_weights = softmax(top_k_logits)`  →  `(B, T, 2)`, cast back to input dtype
   - `top_k_indices`  →  `(B, T, 2)`  (integer indices into the expert list)

3. **Expert computation**: all `num_experts = 4` experts are evaluated:
   - Each `expert_k` is a full `lora.FeedForward` with the same architecture as the
     original action expert FFW (SwiGLU-gated, width 1024, hidden 4096)
   - Results stacked to `(B, T, num_experts, D)`

4. **Weighted aggregation**:
   - One-hot gather selects the top-2 expert outputs  →  `(B, T, 2, D)`
   - Output = `sum over top_k of (weight_k * expert_k_output)`  →  `(B, T, D)`

> **Note**: all N experts are computed regardless of routing, which is standard
> for small N and avoids conditional execution complexity in JAX/XLA.

### Router Z-Loss

To prevent router collapse (all tokens routing to the same expert), an auxiliary
z-loss is added to the training objective:

```
z_loss = mean( log(sum(exp(router_logits)))^2 )
```

This is the ST-MoE router z-loss (Zoph et al., 2022).  It penalises large logit
magnitudes, encouraging the router to stay close to uniform.  It is computed per
layer and averaged across the 18 transformer layers.

---

## Parameter Layout

All parameters inside the transformer scan have a leading depth dimension of 18.
The action expert's FFW subtree (Flax path `PaliGemma/llm/layers/mlp_1/`) looks like:

```
mlp_1/
├── router/
│   └── kernel                      (18, 1024, 4)       trainable
├── expert_0/
│   ├── gating_einsum               (18, 2, 1024, 4096)  frozen
│   ├── linear                      (18, 4096, 1024)     frozen
│   ├── gating_einsum_lora_a        (18, 2, 1024, 32)    trainable
│   ├── gating_einsum_lora_b        (18, 2, 32, 4096)    trainable
│   ├── linear_lora_a               (18, 4096, 32)       trainable
│   └── linear_lora_b               (18, 32, 1024)       trainable
├── expert_1/  (same structure)
├── expert_2/  (same structure)
└── expert_3/  (same structure)
```

### Parameter counts (per layer, one expert)

| Tensor | Shape | Elements |
|---|---|---|
| `gating_einsum` | `(2, 1024, 4096)` | 8,388,608 |
| `linear` | `(4096, 1024)` | 4,194,304 |
| `gating_einsum_lora_a` | `(2, 1024, 32)` | 65,536 |
| `gating_einsum_lora_b` | `(2, 32, 4096)` | 262,144 |
| `linear_lora_a` | `(4096, 32)` | 131,072 |
| `linear_lora_b` | `(32, 1024)` | 32,768 |
| **Expert base total** | | **12,582,912** |
| **Expert LoRA total** | | **491,520** |

With 4 experts × 18 layers:
- Frozen base weights: 4 × 18 × 12,582,912 ≈ **906M parameters**
- Trainable LoRA per expert: 4 × 18 × 491,520 ≈ **35M parameters**
- Router kernel: 18 × 1024 × 4 = **73,728 parameters**
- **Total trainable (MoE): ~35M + router ≈ 35M**

(Compare: pi05_libero_sft trains ~8.9M LoRA params on a single expert.)

---

## Training Configuration (`pi05_libero_moe`)

### MoE Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `num_experts` | 4 | Total experts per layer |
| `top_k` | 2 | Experts activated per token per layer |
| `router_z_loss_coeff` | 1e-3 | Weight of z-loss in total loss |

### Model Architecture

| Parameter | Value |
|---|---|
| `paligemma_variant` | `gemma_2b` |
| `action_expert_variant` | `gemma_300m_lora` (rank=32, alpha=32) |
| `dtype` | `bfloat16` |
| `action_dim` | 32 |
| `action_horizon` | 50 |
| `max_token_len` | 200 |
| `pi05` | True (adaRMSNorm timestep conditioning) |

### Optimizer (defaults from `TrainConfig`)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| `weight_decay` | 1e-10 |
| `clip_gradient_norm` | 1.0 |
| LR schedule | Cosine decay with warmup |
| `peak_lr` | 2.5e-5 |
| `warmup_steps` | 1,000 |
| `decay_steps` | 30,000 |
| `decay_lr` | 2.5e-6 |
| `batch_size` | 32 |
| `num_train_steps` | 30,000 (per task when using `run_sft.sh`) |
| `ema_decay` | None (disabled) |

### Freeze Filter

```
Frozen:
  .*llm.*  (NOT .*llm.*_1.*)        — entire PaliGemma backbone (expert 0)
  .*llm.*_1.*  (NOT .*lora.*)
              (NOT .*router.*)       — action expert base weights

Trainable:
  .*llm.*_1.*lora.*                  — per-expert LoRA adapters (rank 32)
  .*llm.*_1.*router.*                — router kernel (W_router)
  action_in_proj, time_mlp_*, action_out_proj   — projection heads
```

### Weight Initialisation (`MoEWeightLoader`)

Starting from `gs://openpi-assets/checkpoints/pi05_base/params`:

1. For each of the 18 transformer layers, the base action expert FFW weights
   (`mlp_1/gating_einsum` and `mlp_1/linear`) are **copied identically** to all
   four experts (`mlp_1/expert_0/`, …, `mlp_1/expert_3/`).
2. The router kernel `mlp_1/router/kernel` is initialised to **zeros** — this
   makes initial routing uniform (all experts receive equal weight), so the model's
   behaviour at step 0 exactly matches the original pi0.5.
3. All LoRA adapter weights (`lora_a`, `lora_b`) start at **zeros** (consistent
   with standard LoRA initialisation: the LoRA contribution is zero at step 0).

This ensures the model is a strict generalisation of pi0.5 at initialisation.

---

## Training Loss

```
L = L_flow + λ_z * L_router

L_flow  = mean_over_action_horizon( ||v_t - u_t||^2 )
L_router = mean_over_layers( mean_over_tokens( log(sum(exp(router_logits)))^2 ) )
λ_z     = 1e-3
```

- `v_t` is the model's predicted velocity field
- `u_t = noise - clean_actions` is the target velocity
- `router_logits` has shape `(B, T_action, num_experts)` per layer; `T_action = 50`
- z-loss is averaged over 18 layers, then scaled by `λ_z` before being added

During inference (`sample_actions`), the z-loss term is discarded — the router
still activates `top_k` experts, but no auxiliary loss is computed.

---

## File Map

| File | Role |
|---|---|
| `src/openpi/models/moe.py` | `MoEConfig`, `MoEFeedForward`, `MoEBlock`, `MoEModule` |
| `src/openpi/models/pi0_moe.py` | `Pi0MoE` model class (mirrors `pi0.py`) |
| `src/openpi/models/pi0_moe_config.py` | `Pi0MoEConfig` (extends `Pi0Config`) |
| `src/openpi/training/moe_weight_loader.py` | `MoEWeightLoader` (fan-out initialisation) |
| `src/openpi/training/config.py` | `pi05_libero_moe` entry in `_CONFIGS` |

No existing openpi infrastructure files are modified except for the minimal
`_CONFIGS` addition in `config.py`.

---

## Running

```bash
# Standard MoE training run (sequential tasks via sft_train.py):
bash shells/run_sft.sh \
    --config_name pi05_libero_moe \
    --num_tasks 10 \
    --exp_name moe_run_0 \
    --cuda_devices 4,7

# Override MoE hyperparameters via tyro CLI:
bash shells/run_sft.sh \
    --config_name pi05_libero_moe \
    --num_tasks 10 \
    --exp_name moe_8e_run_0 \
    --cuda_devices 4,7 \
    -- --model.moe_config.num_experts 8 --model.moe_config.top_k 2
```
