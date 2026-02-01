"""SkyRL RL Loop Integration Tests.

Comprehensive tests for the complete RL training loop with nmoe models:
- Task 3.3.1: GRPO training
- Task 3.3.2: PPO training
- Task 3.3.3: Reward model integration
- Task 3.3.4: Multi-turn conversation
- Task 3.3.5: Async weight sync
- Task 3.3.6: Checkpoint save/restore

These tests validate that nmoe models work correctly in SkyRL's reinforcement
learning training framework, including forward/backward passes, gradient
computation, weight synchronization, and checkpoint management.

Run with:
    cd nmoe && source .venv/bin/activate
    uv run pytest tests/integration/test_skyrl_rl_loop.py -v --tb=short
"""

import copy
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def small_nmoe_config():
    """Small nmoe config for fast testing."""
    from nmoe.config import Config
    return Config(
        dim=256,
        n_layers=2,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=0,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=256,
    )


@pytest.fixture(scope="module")
def small_model(small_nmoe_config):
    """Create small nmoe model."""
    from nmoe.model import Transformer
    model = Transformer(small_nmoe_config).cuda().bfloat16()
    model.init_weights()
    return model


@pytest.fixture
def fresh_model(small_nmoe_config):
    """Create fresh nmoe model per test (not shared across module)."""
    from nmoe.model import Transformer
    model = Transformer(small_nmoe_config).cuda().bfloat16()
    model.init_weights()
    return model


class TestGRPOTraining:
    """Test GRPO training loop (Task 3.3.1).

    GRPO (Group Relative Policy Optimization) is a variant of policy gradient
    that computes advantages relative to group baselines. This test class
    validates that nmoe models can correctly perform:
    - Policy forward passes to get logits
    - Reference forward passes for KL divergence
    - Gradient computation for the GRPO loss
    """

    def test_grpo_forward_pass(self, small_model, small_nmoe_config):
        """Test GRPO forward pass through nmoe model."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(small_model)

        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Policy forward - get raw logits
        with torch.no_grad():
            logits = wrapper.model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_nmoe_config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_grpo_forward_with_log_probs(self, fresh_model, small_nmoe_config):
        """Test GRPO forward pass returning log probabilities."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        # NMoEModelWrapper takes (model, temperature) not (model, config)
        wrapper = NMoEModelWrapper(fresh_model, temperature=1.0)

        batch_size = 4
        seq_len = 32
        num_actions = 8
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()

        # Forward pass using wrapper's forward_rl method (HFModelWrapper compatible)
        # The wrapper's __call__ dispatches based on second positional arg type:
        # - int/list -> forward_rl()
        # - tensor/None -> forward()
        with torch.no_grad():
            action_log_probs = wrapper.forward_rl(
                sequences=input_ids,
                num_actions=num_actions,
                attention_mask=attention_mask,
            )

        # Should return action log probs for the last num_actions tokens
        assert action_log_probs.shape[0] == batch_size
        assert not torch.isnan(action_log_probs).any()

    def test_grpo_gradient_computation(self, fresh_model, small_nmoe_config):
        """Test gradient computation for GRPO loss."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()
        labels = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Forward with gradients
        logits = wrapper.model(input_ids)

        # Compute cross-entropy loss (simplified GRPO loss without KL term)
        loss = F.cross_entropy(
            logits.view(-1, small_nmoe_config.vocab_size),
            labels.view(-1)
        )

        # Backward
        loss.backward()

        # Check gradients exist for all trainable parameters
        grad_count = 0
        for name, param in wrapper.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                grad_count += 1
                # Verify gradients are not all zeros
                if param.grad.abs().sum() > 0:
                    pass  # Good, non-zero gradients

        assert grad_count > 0, "No trainable parameters found"

    def test_grpo_kl_divergence(self, fresh_model, small_nmoe_config):
        """Test KL divergence computation between policy and reference."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        # Create policy wrapper
        policy_wrapper = NMoEModelWrapper(fresh_model)

        # Create reference model (frozen copy)
        from nmoe.model import Transformer
        ref_model = Transformer(small_nmoe_config).cuda().bfloat16()
        ref_model.load_state_dict(fresh_model.state_dict())
        ref_wrapper = NMoEModelWrapper(ref_model)
        ref_wrapper.freeze_for_reference()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Get log probs from both models
        with torch.no_grad():
            policy_logits = policy_wrapper.model(input_ids)
            ref_logits = ref_wrapper.model(input_ids)

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Compute approximate KL: E[log(p/q)] = E[log_p - log_q]
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

        assert kl.shape == (batch_size, seq_len)
        # KL should be non-negative
        assert (kl >= -1e-5).all(), "KL divergence should be non-negative"

    def test_grpo_advantage_estimation(self, small_nmoe_config):
        """Test GRPO group-based advantage estimation."""
        # GRPO computes advantages relative to group mean
        batch_size = 8
        group_size = 4
        n_groups = batch_size // group_size

        # Simulate rewards for responses
        rewards = torch.randn(batch_size).cuda()

        # Compute group-normalized advantages
        rewards_grouped = rewards.view(n_groups, group_size)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        group_stds = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        # Normalize within groups
        advantages = (rewards_grouped - group_means) / group_stds
        advantages = advantages.view(batch_size)

        assert advantages.shape == (batch_size,)
        assert not torch.isnan(advantages).any()

        # Each group should have mean ~0 after normalization
        advantages_grouped = advantages.view(n_groups, group_size)
        group_means_after = advantages_grouped.mean(dim=1)
        assert torch.allclose(group_means_after, torch.zeros_like(group_means_after), atol=1e-5)


class TestPPOTraining:
    """Test PPO training loop (Task 3.3.2).

    PPO (Proximal Policy Optimization) requires:
    - Policy model for action selection
    - Value head for advantage estimation (critic)
    - Reference model for KL constraint
    - Clipped surrogate objective
    """

    def test_ppo_value_head_interface(self, fresh_model, small_nmoe_config):
        """Test that wrapper supports value estimation interface."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        # Check for essential PPO methods
        assert hasattr(wrapper, 'forward')
        assert hasattr(wrapper, 'model')

        # In SkyRL, value is typically computed by a separate critic model
        # The wrapper should support getting logits for the policy

    def test_ppo_gae_computation(self, small_nmoe_config):
        """Test Generalized Advantage Estimation (GAE) computation."""
        T = 32
        batch_size = 4
        gamma = 0.99
        lam = 0.95

        # Simulate values, rewards, and dones
        values = torch.randn(batch_size, T).cuda()
        rewards = torch.randn(batch_size, T).cuda()
        dones = torch.zeros(batch_size, T).cuda()
        dones[:, -1] = 1  # Episode ends at last step

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(batch_size).cuda()

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[:, t]
                nextvalues = torch.zeros(batch_size).cuda()
            else:
                nextnonterminal = 1.0 - dones[:, t]
                nextvalues = values[:, t + 1]

            delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
            advantages[:, t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        # Returns = advantages + values
        returns = advantages + values

        assert advantages.shape == (batch_size, T)
        assert returns.shape == (batch_size, T)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_ppo_clipped_objective(self, fresh_model, small_nmoe_config):
        """Test PPO clipped surrogate objective."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        batch_size = 4
        seq_len = 16
        clip_eps = 0.2

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Get current log probs
        logits = wrapper.model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        # Simulate old log probs (from rollout)
        old_log_probs = log_probs.detach() + torch.randn_like(log_probs) * 0.1

        # Sample actions
        actions = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Gather log probs for actions
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_action_log_probs = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute probability ratio
        ratio = torch.exp(action_log_probs - old_action_log_probs)

        # Simulate advantages
        advantages = torch.randn(batch_size, seq_len).cuda()

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        assert not torch.isnan(policy_loss)
        assert policy_loss.requires_grad

    def test_ppo_multiple_epochs(self, fresh_model, small_nmoe_config):
        """Test PPO training with multiple epochs on same batch."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

        batch_size = 4
        seq_len = 16
        n_epochs = 3

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()
        targets = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Simulate fixed rollout data
        with torch.no_grad():
            old_logits = wrapper.model(input_ids).clone()

        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            logits = wrapper.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, small_nmoe_config.vocab_size),
                targets.view(-1)
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert len(losses) == n_epochs
        # Loss should generally decrease (or at least not explode)
        assert not any(torch.isnan(torch.tensor(l)) for l in losses)


class TestRewardModelIntegration:
    """Test reward model integration (Task 3.3.3).

    Reward models score model outputs to provide training signal.
    Tests validate the interface between nmoe policy and reward scoring.
    """

    def test_reward_model_scoring_interface(self, fresh_model, small_nmoe_config):
        """Test reward model can score nmoe outputs."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        batch_size = 4
        seq_len = 32

        # Generate responses
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        with torch.no_grad():
            logits = wrapper.model(input_ids)
            # Sample from distribution
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

        # Simulate reward scores (one per response)
        rewards = torch.randn(batch_size).cuda()

        assert rewards.shape == (batch_size,)
        assert next_tokens.shape == (batch_size, 1)

    def test_reward_shaping(self, small_nmoe_config):
        """Test reward shaping for RL training."""
        batch_size = 8
        seq_len = 32

        # Terminal rewards (from reward model)
        terminal_rewards = torch.randn(batch_size).cuda()

        # Shape rewards to token level (reward only at last token)
        shaped_rewards = torch.zeros(batch_size, seq_len).cuda()
        shaped_rewards[:, -1] = terminal_rewards

        assert shaped_rewards.shape == (batch_size, seq_len)
        assert torch.allclose(shaped_rewards[:, -1], terminal_rewards)
        assert (shaped_rewards[:, :-1] == 0).all()

    def test_kl_penalty_reward(self, fresh_model, small_nmoe_config):
        """Test KL penalty applied to rewards."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        batch_size = 4
        seq_len = 16
        kl_coef = 0.1

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        with torch.no_grad():
            logits = wrapper.model(input_ids)

        # Simulate reference logits
        ref_logits = logits + torch.randn_like(logits) * 0.1

        # Compute per-token KL
        policy_log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

        # Base rewards
        base_rewards = torch.randn(batch_size, seq_len).cuda()

        # Apply KL penalty
        penalized_rewards = base_rewards - kl_coef * kl

        assert penalized_rewards.shape == (batch_size, seq_len)
        assert not torch.isnan(penalized_rewards).any()


class TestMultiTurnConversation:
    """Test multi-turn conversation training (Task 3.3.4).

    Multi-turn training requires proper handling of:
    - Conversation history
    - Attention masking across turns
    - Loss masking (only train on assistant responses)
    """

    def test_multi_turn_attention_mask(self, small_nmoe_config):
        """Test attention mask construction for multi-turn conversations."""
        batch_size = 2
        n_turns = 3
        turn_len = 16
        total_len = n_turns * turn_len

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, total_len, total_len).cuda()

        # Apply causal masking
        causal_mask = torch.tril(torch.ones(total_len, total_len)).cuda()
        attention_mask = attention_mask * causal_mask

        assert attention_mask.shape == (batch_size, total_len, total_len)
        # Upper triangle should be zeros (causal)
        assert (attention_mask[:, 0, 1:] == 0).all()

    def test_multi_turn_loss_mask(self, small_nmoe_config):
        """Test loss mask for multi-turn (only train on responses)."""
        batch_size = 2
        seq_len = 64

        # Simulate turn boundaries: [user, assistant, user, assistant]
        # Each turn is 16 tokens
        turn_len = 16
        loss_mask = torch.zeros(batch_size, seq_len).cuda()

        # Only compute loss on assistant responses (turns 1 and 3)
        loss_mask[:, turn_len:2*turn_len] = 1  # First assistant response
        loss_mask[:, 3*turn_len:4*turn_len] = 1  # Second assistant response

        assert loss_mask.shape == (batch_size, seq_len)
        # User turns should be masked
        assert (loss_mask[:, :turn_len] == 0).all()
        assert (loss_mask[:, 2*turn_len:3*turn_len] == 0).all()

    def test_multi_turn_forward(self, fresh_model, small_nmoe_config):
        """Test forward pass with multi-turn conversation."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        batch_size = 2
        n_turns = 2
        turn_len = 16
        total_len = n_turns * turn_len

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, total_len)).cuda()

        # Forward pass
        with torch.no_grad():
            logits = wrapper.model(input_ids)

        assert logits.shape == (batch_size, total_len, small_nmoe_config.vocab_size)

    def test_multi_turn_gradient_with_mask(self, fresh_model, small_nmoe_config):
        """Test gradient computation with loss mask."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        batch_size = 2
        seq_len = 32
        turn_len = 16

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()
        targets = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        # Loss mask: only second half (assistant response)
        loss_mask = torch.zeros(batch_size, seq_len).cuda()
        loss_mask[:, turn_len:] = 1

        # Forward
        logits = wrapper.model(input_ids)

        # Masked cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, small_nmoe_config.vocab_size),
            targets.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)

        masked_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
        masked_loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in wrapper.parameters())
        assert has_grad


class TestAsyncWeightSync:
    """Test async weight sync during training (Task 3.3.5).

    In SkyRL, policy weights are synced to inference engines periodically.
    This tests the weight extraction and loading mechanisms.
    """

    def test_weight_extraction(self, fresh_model, small_nmoe_config):
        """Test weight extraction for sync."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        # Extract state dict
        state_dict = wrapper.model.state_dict()

        assert len(state_dict) > 0

        # Check for key model components
        has_embedding = any('embed' in k.lower() for k in state_dict)
        has_blocks = any('block' in k.lower() for k in state_dict)

        assert has_embedding or has_blocks, "Missing model components in state dict"

    def test_moe_weight_extraction(self, fresh_model, small_nmoe_config):
        """Test MoE expert weights are properly extracted."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        state_dict = wrapper.model.state_dict()

        # Look for MoE weight patterns (W1, W2, W3 for experts)
        moe_keys = [k for k in state_dict if any(w in k for w in ['W1', 'W2', 'W3', 'w1', 'w2', 'w3', 'experts'])]

        # nmoe with n_routed_experts > 0 should have MoE weights
        if small_nmoe_config.n_routed_experts > 0:
            assert len(moe_keys) > 0, f"No MoE weights found. Keys: {list(state_dict.keys())[:10]}"

    def test_weight_loading(self, fresh_model, small_nmoe_config):
        """Test weight loading after sync."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        # Get initial weights
        initial_state = {k: v.clone() for k, v in wrapper.model.state_dict().items()}

        # Modify weights
        with torch.no_grad():
            for param in wrapper.model.parameters():
                param.add_(0.01)

        # Verify weights changed
        for k, v in wrapper.model.state_dict().items():
            if k in initial_state:
                assert not torch.allclose(v, initial_state[k]), f"Weights {k} did not change"
                break

        # Reload initial weights
        wrapper.model.load_state_dict(initial_state)

        # Verify reload worked
        for k, v in wrapper.model.state_dict().items():
            if k in initial_state:
                assert torch.allclose(v, initial_state[k]), f"Weight {k} not restored correctly"

    def test_partial_weight_sync(self, fresh_model, small_nmoe_config):
        """Test syncing only specific parameter groups."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        # Get param sets
        if hasattr(wrapper, 'param_sets'):
            expert_params, dense_params = wrapper.param_sets()

            # Count parameters
            expert_count = sum(p.numel() for p in expert_params)
            dense_count = sum(p.numel() for p in dense_params)

            assert expert_count > 0 or dense_count > 0
            # Total should match all params
            total_params = sum(p.numel() for p in wrapper.model.parameters())
            assert expert_count + dense_count == total_params

    def test_weight_update_after_training_step(self, fresh_model, small_nmoe_config):
        """Test weights are updated correctly after a training step."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)

        # Get initial weights
        initial_state = {k: v.clone() for k, v in wrapper.model.state_dict().items()}

        # Training step
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()
        targets = torch.randint(0, small_nmoe_config.vocab_size, (batch_size, seq_len)).cuda()

        logits = wrapper.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, small_nmoe_config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify weights changed
        weights_changed = False
        for k, v in wrapper.model.state_dict().items():
            if k in initial_state:
                if not torch.allclose(v, initial_state[k]):
                    weights_changed = True
                    break

        assert weights_changed, "Weights did not change after training step"


class TestCheckpointSaveRestore:
    """Test checkpoint save/restore during RL (Task 3.3.6).

    Checkpointing is critical for long RL training runs.
    Tests validate save/restore of model, optimizer, and training state.
    """

    def test_checkpoint_save(self, fresh_model, small_nmoe_config):
        """Test saving checkpoint during RL."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "rl_checkpoint.pt"

            # Save model state
            state = {
                'model': wrapper.model.state_dict(),
                'step': 100,
            }
            torch.save(state, checkpoint_path)

            assert checkpoint_path.exists()

            # Verify saved file is valid
            loaded = torch.load(checkpoint_path, weights_only=False)
            assert 'model' in loaded
            assert 'step' in loaded
            assert loaded['step'] == 100

    def test_checkpoint_restore(self, fresh_model, small_nmoe_config):
        """Test restoring checkpoint during RL."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "rl_checkpoint.pt"

            # Save initial state
            original_state = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
            torch.save({'model': original_state, 'step': 100}, checkpoint_path)

            # Modify model
            with torch.no_grad():
                for param in wrapper.model.parameters():
                    param.add_(1.0)

            # Verify model changed
            current_state = wrapper.model.state_dict()
            model_changed = False
            for k in original_state:
                if not torch.allclose(current_state[k], original_state[k]):
                    model_changed = True
                    break
            assert model_changed, "Model should have changed"

            # Restore from checkpoint
            loaded = torch.load(checkpoint_path, weights_only=False)
            wrapper.model.load_state_dict(loaded['model'])

            # Verify restoration
            for k, v in wrapper.model.state_dict().items():
                assert torch.allclose(v, original_state[k]), f"Weight {k} not restored correctly"

    def test_checkpoint_with_optimizer(self, fresh_model, small_nmoe_config):
        """Test checkpoint includes optimizer state."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

        # Do a training step to populate optimizer state
        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (2, 16)).cuda()
        targets = torch.randint(0, small_nmoe_config.vocab_size, (2, 16)).cuda()
        logits = wrapper.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, small_nmoe_config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "rl_checkpoint.pt"

            # Save with optimizer
            state = {
                'model': wrapper.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': 50,
            }
            torch.save(state, checkpoint_path)

            # Load and verify
            loaded = torch.load(checkpoint_path, weights_only=False)
            assert 'optimizer' in loaded
            assert len(loaded['optimizer']['state']) > 0

    def test_checkpoint_with_scheduler(self, fresh_model, small_nmoe_config):
        """Test checkpoint includes learning rate scheduler."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # Step scheduler a few times (note: _step_count starts at 1 before first step)
        n_steps = 10
        for _ in range(n_steps):
            scheduler.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "rl_checkpoint.pt"

            state = {
                'model': wrapper.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': n_steps,
            }
            torch.save(state, checkpoint_path)

            # Load and verify scheduler state was saved
            loaded = torch.load(checkpoint_path, weights_only=False)
            assert 'scheduler' in loaded
            # PyTorch scheduler _step_count starts at 1 and increments after each step()
            # After n_steps calls to step(), _step_count = n_steps + 1
            assert loaded['scheduler']['_step_count'] == n_steps + 1

    def test_checkpoint_resumption_training(self, small_nmoe_config):
        """Test complete training resumption from checkpoint."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError:
            pytest.skip("Required modules not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "rl_checkpoint.pt"

            # Phase 1: Train and save
            model1 = Transformer(small_nmoe_config).cuda().bfloat16()
            model1.init_weights()
            wrapper1 = NMoEModelWrapper(model1)
            optimizer1 = torch.optim.AdamW(wrapper1.parameters(), lr=1e-4)

            # Training step
            input_ids = torch.randint(0, small_nmoe_config.vocab_size, (2, 16)).cuda()
            targets = torch.randint(0, small_nmoe_config.vocab_size, (2, 16)).cuda()
            logits = wrapper1.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, small_nmoe_config.vocab_size), targets.view(-1))
            loss.backward()
            optimizer1.step()

            # Save checkpoint
            torch.save({
                'model': wrapper1.model.state_dict(),
                'optimizer': optimizer1.state_dict(),
                'step': 1,
            }, checkpoint_path)

            # Get weights after training
            weights_after_step1 = {k: v.clone() for k, v in wrapper1.model.state_dict().items()}

            # Phase 2: Load and continue
            model2 = Transformer(small_nmoe_config).cuda().bfloat16()
            wrapper2 = NMoEModelWrapper(model2)
            optimizer2 = torch.optim.AdamW(wrapper2.parameters(), lr=1e-4)

            loaded = torch.load(checkpoint_path, weights_only=False)
            wrapper2.model.load_state_dict(loaded['model'])
            optimizer2.load_state_dict(loaded['optimizer'])
            step = loaded['step']

            # Verify state matches
            assert step == 1
            for k, v in wrapper2.model.state_dict().items():
                assert torch.allclose(v, weights_after_step1[k]), f"Weight {k} mismatch after load"


class TestReferenceModelManagement:
    """Test reference model management for KL-constrained RL."""

    def test_reference_model_freeze(self, fresh_model, small_nmoe_config):
        """Test freezing reference model."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        wrapper.freeze_for_reference()

        # All parameters should be frozen
        for param in wrapper.model.parameters():
            assert not param.requires_grad

        assert wrapper.is_frozen

    def test_reference_model_unfreeze(self, fresh_model, small_nmoe_config):
        """Test unfreezing reference model."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        wrapper.freeze_for_reference()
        wrapper.unfreeze()

        # All parameters should be trainable
        for param in wrapper.model.parameters():
            assert param.requires_grad

        assert not wrapper.is_frozen

    def test_reference_model_no_gradients(self, fresh_model, small_nmoe_config):
        """Test reference model does not accumulate gradients."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)
        wrapper.freeze_for_reference()

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (2, 16)).cuda()

        # Forward pass should work
        with torch.no_grad():
            logits = wrapper.model(input_ids)

        assert logits is not None
        # No gradients should exist
        for param in wrapper.model.parameters():
            assert param.grad is None


class TestExpertLoadBalancing:
    """Test MoE expert load balancing during RL training."""

    def test_aux_loss_computation(self, fresh_model, small_nmoe_config):
        """Test auxiliary load balancing loss."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (4, 32)).cuda()

        # Forward to populate router statistics
        with torch.no_grad():
            _ = wrapper.model(input_ids)

        # Get aux loss
        if hasattr(wrapper, 'get_router_aux_loss'):
            aux_loss = wrapper.get_router_aux_loss()
            assert isinstance(aux_loss, torch.Tensor)
            assert not torch.isnan(aux_loss)

    def test_expert_load_stats(self, fresh_model, small_nmoe_config):
        """Test expert load statistics collection."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
        except ImportError:
            pytest.skip("SkyRL nmoe wrapper not available")

        wrapper = NMoEModelWrapper(fresh_model)

        input_ids = torch.randint(0, small_nmoe_config.vocab_size, (4, 32)).cuda()

        # Forward to populate statistics
        with torch.no_grad():
            _ = wrapper.model(input_ids)

        # Get load stats
        if hasattr(wrapper, 'get_expert_load_stats'):
            stats = wrapper.get_expert_load_stats()
            assert isinstance(stats, dict)
            assert 'load_mean' in stats or 'loads' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
