"""
Lightweight AVA-VLA component checks without constructing a full OpenVLA model.
"""

from pathlib import Path
import os
import sys

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from prismatic.models.vlas.avavla import ExitGate, LatentTransition, ReasoningPolicy, ValueFunction


def test_core_components() -> bool:
    batch_size, latent_dim, obs_dim, update_dim = 4, 128, 128, 32
    policy = ReasoningPolicy(
        latent_dim,
        obs_dim,
        hidden_dim=128,
        update_dim=update_dim,
        num_heads=4,
        num_layers=1,
    )
    transition = LatentTransition(
        latent_dim,
        obs_dim,
        hidden_dim=128,
        update_dim=update_dim,
        num_heads=4,
    )
    exit_gate = ExitGate(latent_dim, hidden_dim=128)
    value_fn = ValueFunction(latent_dim, hidden_dim=128)

    z_t = torch.randn(batch_size, latent_dim)
    o_t = torch.randn(batch_size, obs_dim)
    policy_output = policy(z_t, o_t)
    update_action, log_prob, entropy = policy.sample_update_action(policy_output, training=True)
    z_next = transition(z_t, o_t, update_action)
    exit_score = exit_gate(z_next)
    value = value_fn(z_next)

    assert policy_output["logits"].shape == (batch_size, update_dim)
    assert policy_output["probs"].shape == (batch_size, update_dim)
    assert torch.allclose(policy_output["probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert update_action.shape == (batch_size, update_dim)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert z_next.shape == (batch_size, latent_dim)
    assert exit_score.shape == (batch_size, 1)
    assert value.shape == (batch_size, 1)
    assert torch.isfinite(z_next).all()
    print("PASS: Softmax policy, GRU transition, exit gate, and value function are shape-valid.")
    return True


def test_policy_ratio_updates() -> bool:
    batch_size, latent_dim, obs_dim, update_dim = 4, 64, 64, 16
    policy = ReasoningPolicy(
        latent_dim,
        obs_dim,
        hidden_dim=64,
        update_dim=update_dim,
        num_heads=4,
        num_layers=1,
    )
    z_t = torch.randn(batch_size, latent_dim)
    o_t = torch.randn(batch_size, obs_dim)

    with torch.no_grad():
        old_output = policy(z_t, o_t)
        update_action, old_log_prob, _ = policy.sample_update_action(old_output, training=True)
        update_action = update_action.detach()
        old_log_prob = old_log_prob.detach()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    for _ in range(3):
        output = policy(z_t, o_t)
        log_prob, _ = policy.evaluate_update_action(output, update_action)
        loss = -log_prob.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    new_output = policy(z_t, o_t)
    new_log_prob, _ = policy.evaluate_update_action(new_output, update_action)
    ratio = torch.exp((new_log_prob - old_log_prob).clamp(-20.0, 20.0))
    assert torch.isfinite(ratio).all()
    assert not torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)
    print("PASS: Stored old log-probs produce nontrivial PPO ratios after a policy update.")
    return True


def main() -> int:
    return 0 if test_core_components() and test_policy_ratio_updates() else 1


if __name__ == "__main__":
    raise SystemExit(main())
