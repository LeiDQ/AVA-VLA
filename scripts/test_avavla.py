"""
test_avavla.py

Simple test script to verify AVA-VLA implementation.
"""

import torch
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))


def test_imports():
    """Test if all modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from prismatic.models.vlas.avavla import (
            AVAVLA,
            ReasoningPolicy,
            LatentTransition,
            ExitGate,
            ValueFunction,
        )
        print("✓ Successfully imported AVA-VLA components")
    except Exception as e:
        print(f"✗ Failed to import AVA-VLA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from vla_scripts.deploy_avavla import load_avavla_model, predict_action
        print("✓ Successfully imported deployment utilities")
    except Exception as e:
        print(f"✗ Failed to import deployment utilities: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_components():
    """Test individual components."""
    print("\n" + "=" * 60)
    print("Testing individual components...")
    print("=" * 60)
    
    from prismatic.models.vlas.avavla import (
        ReasoningPolicy,
        LatentTransition,
        ExitGate,
        ValueFunction,
    )
    
    batch_size = 4
    latent_dim = 512
    obs_dim = 768
    device = "cpu"
    
    # Test ReasoningPolicy
    try:
        policy = ReasoningPolicy(latent_dim, obs_dim)
        z_t = torch.randn(batch_size, latent_dim)
        o_t = torch.randn(batch_size, obs_dim)
        policy_output = policy(z_t, o_t)
        update_action, log_prob, entropy = policy.sample_update_action(policy_output, training=True)
        assert policy_output["logits"].shape == (batch_size, 64)
        assert policy_output["probs"].shape == (batch_size, 64)
        assert update_action.shape == (batch_size, 64)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()
        assert torch.allclose(policy_output["probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)
        print(f"✓ ReasoningPolicy: Softmax logits/probs shape {policy_output['logits'].shape}")
    except Exception as e:
        print(f"✗ ReasoningPolicy failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test LatentTransition
    try:
        transition = LatentTransition(latent_dim, obs_dim)
        u_t = torch.randn(batch_size, 64)
        z_next = transition(z_t, o_t, u_t)
        assert z_next.shape == (batch_size, latent_dim)
        print(f"✓ LatentTransition: output shape {z_next.shape}")
    except Exception as e:
        print(f"✗ LatentTransition failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test ExitGate
    try:
        exit_gate = ExitGate(latent_dim)
        e_t = exit_gate(z_t)
        assert e_t.shape == (batch_size, 1)
        assert (e_t >= 0).all() and (e_t <= 1).all()
        print(f"✓ ExitGate: output shape {e_t.shape}, range [0, 1]")
    except Exception as e:
        print(f"✗ ExitGate failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test ValueFunction
    try:
        value_fn = ValueFunction(latent_dim)
        v_t = value_fn(z_t)
        assert v_t.shape == (batch_size, 1)
        print(f"✓ ValueFunction: output shape {v_t.shape}")
    except Exception as e:
        print(f"✗ ValueFunction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_reasoning_loop():
    """Test reasoning loop."""
    print("\n" + "=" * 60)
    print("Testing reasoning loop...")
    print("=" * 60)
    
    from prismatic.models.vlas.avavla import (
        AVAVLA,
        ReasoningPolicy,
        LatentTransition,
        ExitGate,
    )
    
    batch_size = 2
    latent_dim = 512
    obs_dim = 768
    max_steps = 5
    exit_threshold = 0.8
    device = "cpu"
    
    try:
        # Create components
        policy = ReasoningPolicy(latent_dim, obs_dim)
        transition = LatentTransition(latent_dim, obs_dim)
        exit_gate = ExitGate(latent_dim)
        
        # Initialize
        z_t = torch.randn(batch_size, latent_dim)
        o_t = torch.randn(batch_size, obs_dim)
        
        # Simulate reasoning loop
        current_z = z_t.clone()
        exit_scores = []
        
        for step in range(max_steps):
            policy_output = policy(current_z, o_t)
            u_t, _, _ = policy.sample_update_action(policy_output, training=False)
            current_z = transition(current_z, o_t, u_t)
            e_t = exit_gate(current_z)
            exit_scores.append(e_t)
            
            if (e_t.squeeze(-1) > exit_threshold).any():
                break
        
        exit_scores = torch.cat(exit_scores, dim=1)
        
        print(f"✓ Reasoning loop completed")
        print(f"  - Steps performed: {exit_scores.shape[1]}")
        print(f"  - Exit scores shape: {exit_scores.shape}")
        print(f"  - Final latent state shape: {current_z.shape}")
        
    except Exception as e:
        print(f"✗ Reasoning loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_rl_loss_computation():
    """Test RL loss computation."""
    print("\n" + "=" * 60)
    print("Testing RL loss computation...")
    print("=" * 60)
    
    from prismatic.models.vlas.avavla import AVAVLA, ExitGate, ReasoningPolicy, ValueFunction
    
    batch_size = 2
    num_steps = 3
    latent_dim = 64
    obs_dim = 64
    update_dim = 16
    
    try:
        avavla = type("DummyAVAVLA", (), {})()
        avavla.enable_latent_reasoning = True
        avavla.reasoning_policy = ReasoningPolicy(
            latent_dim,
            obs_dim,
            hidden_dim=64,
            update_dim=update_dim,
            num_heads=4,
            num_layers=1,
        )
        avavla.value_function = ValueFunction(latent_dim, hidden_dim=64)
        avavla.exit_gate = ExitGate(latent_dim, hidden_dim=64)
        
        # Simulate trajectory
        latent_states = torch.randn(batch_size, num_steps, latent_dim)
        obs_encodings = torch.randn(batch_size, obs_dim)
        obs_steps = obs_encodings.unsqueeze(1).expand(-1, num_steps, -1)
        flat_policy_output = avavla.reasoning_policy(
            latent_states.reshape(batch_size * num_steps, -1),
            obs_steps.reshape(batch_size * num_steps, -1),
        )
        update_actions, old_log_probs, entropies = avavla.reasoning_policy.sample_update_action(
            flat_policy_output,
            training=True,
        )
        update_actions = update_actions.reshape(batch_size, num_steps, update_dim).detach()
        old_log_probs = old_log_probs.reshape(batch_size, num_steps).detach()
        entropies = entropies.reshape(batch_size, num_steps).detach()
        next_latent_states = latent_states + 0.01 * torch.randn_like(latent_states)
        exit_scores = avavla.exit_gate(next_latent_states.reshape(batch_size * num_steps, -1)).reshape(
            batch_size,
            num_steps,
        )
        rewards = torch.rand(batch_size)
        trajectories = {
            "latent_states": latent_states,
            "next_latent_states": next_latent_states,
            "update_actions": update_actions,
            "action_log_probs": old_log_probs,
            "old_action_log_probs": old_log_probs,
            "action_entropies": entropies,
            "exit_scores": exit_scores,
            "valid_mask": torch.ones(batch_size, num_steps),
            "obs_encodings": obs_encodings,
        }
        rl_loss, rl_info = AVAVLA.compute_rl_loss(avavla, trajectories, rewards)
        
        print(f"✓ RL loss computation successful")
        print(f"  - Total RL loss: {rl_loss.item():.4f}")
        print(f"  - PPO ratio mean: {rl_info['ppo_ratio_mean']:.4f}")
        print(f"  - Policy recomputed: {rl_info['ppo_policy_recomputed']:.0f}")
        assert torch.isfinite(rl_loss)
        assert rl_info["ppo_policy_recomputed"] == 1.0
        
    except Exception as e:
        print(f"✗ RL loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_file_structure():
    """Test if all required files exist."""
    print("\n" + "=" * 60)
    print("Testing file structure...")
    print("=" * 60)
    
    required_files = [
        "prismatic/models/vlas/__init__.py",
        "prismatic/models/vlas/avavla.py",
        "vla-scripts/finetune_avavla.py",
        "vla-scripts/deploy_avavla.py",
        "scripts/evaluate_avavla.py",
        "AVA_VLA_README.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AVA-VLA Implementation Test Suite")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Test file structure
    results['file_structure'] = test_file_structure()
    
    # Test imports
    results['imports'] = test_imports()
    
    if results['imports']:
        # Test individual components
        results['components'] = test_components()
        
        # Test reasoning loop
        results['reasoning_loop'] = test_reasoning_loop()
        
        # Test RL loss computation
        results['rl_loss'] = test_rl_loss_computation()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
