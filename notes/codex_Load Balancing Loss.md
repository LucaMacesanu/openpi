We observe severe MoE routing collapse: moe/global_expert_0_usage is 100%, while all other experts have near-zero usage.

Please inspect the current MoE implementation and identify what router-related losses are implemented.

Specifically:
1. Is there only router z-loss?
2. Is there a load balancing / auxiliary expert usage loss?
3. If load balancing loss is missing, add one suitable for top-1 routing.

Goal:
Encourage balanced expert utilization and prevent all tokens from routing to expert_0.

Implementation requirements:
- Keep existing router z-loss.
- Add a load-balancing loss based on expert usage and router probabilities.
- Log the new loss separately as moe/load_balance_loss.
- Add a configurable coefficient, e.g. load_balance_loss_weight.
- Do not change inference behavior.
- Make sure total loss = flow_loss + z_loss_weight * router_z_loss + load_balance_loss_weight * load_balance_loss.

Also suggest reasonable starting values for:
- router_z_loss_weight
- load_balance_loss_weight

Current symptom:
moe/global_expert_0_usage = 100%