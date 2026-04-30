Goal:
Implement a Residual MoE variant for pi0 models.

This is NOT replacement MoE.
This is Residual MoE with one always-on large FFN and one routed small expert.

--------------------------------------------------
Core Design (IMPORTANT)
--------------------------------------------------

For each token:

  base = large_dense_ffn(x)              # always active
  small = selected_small_expert(x)       # top-1 from small experts
  output = base + residual_scale * small

Key points:
- The large FFN is ALWAYS used.
- The router ONLY selects among the small experts.
- We are NOT choosing between large and small.
- We are combining them.

--------------------------------------------------
Model Architecture
--------------------------------------------------

1. Large FFN (base path)
- Same as original action expert FFN
- Shape: 1024 → 4096 → 1024
- Loaded from original dense checkpoint
- Always active

2. Small experts (MoE residual branch)
- Number of experts: 8
- Each expert is smaller:
  1024 → 128 → 1024 (or 128 bottleneck then project back to 1024)
- Router selects top_k = 1 expert

3. Residual combination
  output = base + residual_scale * delta

- residual_scale should be configurable (default = 1.0)

--------------------------------------------------
Routing (IMPORTANT)
--------------------------------------------------

- Routing is ONLY applied to small experts
- Use top_k = 1
- Router operates on hidden state x
- Use softmax(router_logits) before argmax

--------------------------------------------------
Losses
--------------------------------------------------

1. Keep existing:
- flow loss
- router z-loss (if already implemented)

2. Add load balancing loss (for small experts only)
- Switch-style load balancing loss:
  based on:
    - hard expert usage
    - softmax probability mean
- Add config:
    load_balance_loss_weight (default = 1e-2)

3. Total loss:

  total_loss =
      flow_loss
    + router_z_loss_weight * router_z_loss
    + load_balance_loss_weight * load_balance_loss

--------------------------------------------------
Initialization (VERY IMPORTANT)
--------------------------------------------------

- Large FFN: load from dense checkpoint
- Small experts: random init
- Residual branch should start near zero:

  Option 1 (preferred):
    initialize final projection of small experts near zero

  Option 2:
    residual_scale small (e.g. 0.1)

Goal:
- Step-0 behavior should be close to dense model

--------------------------------------------------
Logging (wandb)
--------------------------------------------------

Log MoE diagnostics for small experts only:

1. Expert usage
- moe/global_expert_{k}_usage
- moe/layer_{i}/expert_{k}_usage

2. Router distribution stats
- moe/router_entropy
- moe/router_prob_variance
- moe/router_prob_mean

3. Router logits stats
- moe/router_logits_mean
- moe/router_logits_std
- moe/router_logits_max

4. Loss terms
- moe/load_balance_loss
- moe/router_z_loss

Do NOT log large tensors.

--------------------------------------------------
Code Structure
--------------------------------------------------

- Add new module:
    ResidualMoE (in moe.py or new file)

- Do NOT break existing:
    pi0_moe
    pi05_moe

- Add new configs:
    pi0_residual_moe

- Update training config to support new preset:
    pi0_libero_residual_moe

--------------------------------------------------
Constraints
--------------------------------------------------

- Do NOT change behavior of existing models
- Do NOT modify scripts/train.py
- Compatible with scripts/train_moe.py
- Keep JAX JIT safe
- Keep memory usage reasonable

--------------------------------------------------
Sanity Checks
--------------------------------------------------

1. residual_scale = 0 → identical to dense model
2. small MoE near-zero → close to dense behavior
3. No shape mismatch
4. Training runs without crash
5. Existing MoE configs still work

--------------------------------------------------
Before coding, inspect:
--------------------------------------------------

- src/openpi/models/moe.py
- src/openpi/models/pi0_moe.py
- src/openpi/models/pi05_moe.py
- src/openpi/models/pi0_config.py
- src/openpi/training/config.py
- scripts/train_moe.py

--------------------------------------------------
After implementation, explain:
--------------------------------------------------

- What files were modified
- How Residual MoE forward works
- How checkpoint loading works
- How losses are computed
- What wandb metrics are added