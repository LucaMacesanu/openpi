Goal:
Implement a pi0_libero_moe variant following the same high-level idea as pi05_libero_moe: replace the action expert FFN with an MoE FFN, using MoE(input = hidden) directly.

Context:
Current pi05_libero_moe uses 4 MoE experts, each expert being a 1024-wide action FFN:
4 x (1024 -> 4096 -> 1024) plus a router.
Relevant files:
- openpi/src/openpi/models/pi0.py
- openpi/src/openpi/models/pi0_moe.py
- openpi/src/openpi/models/moe.py
- openpi/src/openpi/models/pi0_config.py
- openpi/src/openpi/models/pi0_moe_config.py
- openpi/src/openpi/training/config.py
- openpi/src/openpi/training/moe_weight_loader.py

Design direction:
For pi0_libero_moe, do NOT try to separate action/state/timestep before routing for now.
Even though pi0 hidden states contain different information from pi0.5 hidden states, use the same simple rule:
MoE(input = hidden)

That means:
- In pi0.5, MoE receives the pi0.5 action-suffix hidden states.
- In pi0, MoE receives the pi0 suffix hidden states as-is, even if those hidden states include action/state/timestep information.
- This is intended as a first baseline / minimal implementation, not the final clean semantic design.

Tasks:
1. Inspect the current pi05 MoE implementation and identify exactly which assumptions are pi0.5-specific.
2. Add or modify the MoE model/config so that pi0_libero_moe supports pi05=False correctly.
3. Preserve the original pi0 suffix behavior:
   - include state_proj(obs.state)
   - include action_time_mlp_in/action_time_mlp_out behavior
   - do not accidentally use the pi0.5-only adaRMS suffix path
4. Replace the action expert FFN with MoE in the same transformer blocks, using hidden as the router/expert input.
5. Add a pi0_libero_moe training preset based on pi0_libero, not pi05_libero.
6. Make checkpoint loading work from pi0_base, not pi05_base.
7. Keep the MoE initialization behavior:
   - clone the original dense action FFN weights into every expert
   - leave the router initialized by the MoE module
   - leave MoE-specific LoRA weights at init if applicable

Important constraints:
- Do not change pi05_libero_moe behavior.
- Do not change dense pi0 or dense pi0.5 behavior.
- Keep action expert width = 1024 and mlp_dim = 4096 unless the existing config says otherwise.
- Use num_experts=4 by default for pi0_libero_moe.
- Use top_k consistent with the existing pi05 MoE config unless there is a strong reason to change it.
- Avoid introducing a new data transform unless required.
- If PI0 and PI05 preprocessing differs, make sure pi0_libero_moe follows PI0 preprocessing.

Sanity checks:
1. Verify that pi0_libero_moe with num_experts=1 and top_k=1 behaves as close as possible to dense pi0.
2. Verify checkpoint key mapping:
   original mlp_1/* -> mlp_1/expert_k/* for each expert k.
3. Verify pi0_libero_moe does not accidentally inherit pi0.5-only fields such as discrete_state_input=True unless explicitly intended.
4. Print or document the trainable parameter groups and confirm they match the intended freeze policy.
5. Add a short note explaining that this baseline uses MoE(input = hidden), so in pi0 the router sees the full suffix hidden representation rather than action-only hidden states.

Done when:
- pi0_libero_moe can be instantiated.
- It can load from a pi0_base checkpoint.
- A small forward/loss pass runs without shape errors.
- Existing pi05_libero_moe still works unchanged.
- The code comments clearly state that this is a minimal baseline and not a semantically aligned pi0/pi0.5 MoE comparison.

Important refactor request:
The current file/class names use pi0_moe, but the implementation is actually pi0.5-specific. Please refactor the MoE code so pi0 MoE and pi0.5 MoE are clearly separated.

Desired structure:
- pi05_moe.py / Pi05MoE: current MoE implementation, preserving existing pi05_libero_moe behavior.
- pi0_moe.py / Pi0MoE: new implementation for true pi0, preserving dense pi0 suffix behavior.
- pi05_moe_config.py / Pi05MoEConfig: pi0.5-specific config.
- pi0_moe_config.py / Pi0MoEConfig: pi0-specific config.

Constraints:
- Do not change existing pi05_libero_moe behavior except renaming/refactoring imports.
- Ensure pi05_moe always uses pi05=True behavior.
- Ensure pi0_moe always uses pi05=False behavior.
- Avoid one generic class with many pi05 conditionals unless absolutely necessary.
- Update training presets so:
  - pi05_libero_moe uses Pi05MoE/Pi05MoEConfig.
  - pi0_libero_moe uses Pi0MoE/Pi0MoEConfig.
- Update checkpoint loading so pi05_moe loads from pi05_base and pi0_moe loads from pi0_base.
- Add comments explaining that pi05 MoE and pi0 MoE differ because their suffix/state/timestep paths differ.