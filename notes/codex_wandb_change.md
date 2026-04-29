We currently log the following metrics to wandb:
- loss
- grad_norm
- param_norm

We are training a Mixture-of-Experts (MoE) model with top_k = 1 (top-1 routing):

uv run scripts/train.py pi0_libero_moe \
  --exp-name=my_pi0_libero_moe_ft_bs64 \
  --checkpoint-base-dir=$SCRATCH/openpi_ckpts \
  --batch-size 64 \
  --resume

I want to extend the logging to include MoE-specific diagnostics.

IMPORTANT:
We are NOT interested in top-k selected weights only.
Instead, we want to analyze the FULL softmax distribution of router probabilities.

--------------------------------------------------
Goals
--------------------------------------------------

We want to understand:
- whether routing is collapsing (always selecting one expert)
- whether routing is random (no meaningful specialization)
- how confident the router is (distribution sharpness)

--------------------------------------------------
Required MoE metrics
--------------------------------------------------

1. Expert usage (REQUIRED)

For each MoE layer:
- count how many tokens are routed to each expert
- normalize into percentage

Log:
- moe/layer_{i}/expert_{k}_usage
- moe/global_expert_{k}_usage

Purpose:
- detect collapse (one expert dominates)
- detect dead experts

--------------------------------------------------

2. Router probability distribution statistics (CRITICAL)

Let:
  probs = softmax(router_logits)

We want to log statistics over the FULL distribution (not just top-1).

Compute:

- mean(probabilities)
- variance(probabilities)

Log:
- moe/router_prob_mean
- moe/router_prob_variance

Purpose:
- variance measures sharpness (confidence)
  - high variance → very confident (possibly collapse)
  - low variance → flat distribution (possibly random)

--------------------------------------------------

3. Router entropy (REQUIRED)

Compute entropy over the same softmax probabilities:

  entropy = -sum(p * log(p))

Log:
- moe/router_entropy

Purpose:
- low entropy → collapse
- high entropy → random
- mid entropy → healthy routing

--------------------------------------------------

4. Router logits statistics

Log:
- moe/router_logits_mean
- moe/router_logits_std
- moe/router_logits_max

If available:
- moe/router_z_loss

Purpose:
- monitor numerical stability
- detect logit explosion

--------------------------------------------------

5. Per-layer diagnostics (IMPORTANT)

For each MoE layer, compute:
- expert usage
- router entropy
- (optional) variance

Log:
- moe/layer_{i}/expert_usage
- moe/layer_{i}/router_entropy

Purpose:
- detect layer-wise collapse (often only happens in deeper layers)

--------------------------------------------------

Implementation constraints
--------------------------------------------------

- Do NOT change model behavior or training logic.
- Only add logging.
- Keep scripts/train.py unchanged.
- Create a new script:
    scripts/train_moe.py

- Follow the same structure:
  - metrics computed inside train_step()
  - returned via info dict
  - aggregated and logged with wandb.log(...)

- Do NOT log large tensors every step.
- Use aggregated scalars or small vectors.

- If MoE code does not expose router_logits or probabilities:
  - minimally modify moe.py to return lightweight summaries
  - do not break JAX JIT

--------------------------------------------------

Sanity requirements
--------------------------------------------------

- Existing metrics (loss, grad_norm, param_norm) must still work
- No noticeable slowdown
- No JAX compilation issues

--------------------------------------------------

Goal summary
--------------------------------------------------

We are NOT just tracking which expert is selected.
We are tracking the FULL routing distribution.

This allows us to distinguish:
- collapse (one-hot distribution)
- random routing (uniform distribution)
- meaningful routing (structured, confident but not degenerate)