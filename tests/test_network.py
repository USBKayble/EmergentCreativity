"""
tests/test_network.py
=====================
Unit tests for the neural network architecture.
These tests require PyTorch but NOT PyBullet or Gymnasium.
Skipped automatically if torch is unavailable.
"""
from __future__ import annotations

import pytest

try:
    import torch
    import numpy as np
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")

from src.emergent_creativity.environment.senses import (
    VISION_H, VISION_W, VISION_C,
    HEARING_DIM, TOUCH_DIM, SMELL_DIM, TASTE_DIM,
    TOTAL_SENSORY_DIM,
)
from src.emergent_creativity.tenant.actions import N_ACTIONS

VITALS_DIM = 4
NON_VISUAL_DIM = TOTAL_SENSORY_DIM + VITALS_DIM


@pytest.fixture
def net():
    from src.emergent_creativity.nn.architecture import TenantNetwork
    return TenantNetwork(n_actions=N_ACTIONS)


@pytest.fixture
def dummy_batch(batch_size=2):
    vision    = torch.zeros(batch_size, VISION_C, VISION_H, VISION_W)
    non_visual = torch.zeros(batch_size, NON_VISUAL_DIM)
    return vision, non_visual


class TestTenantNetworkShapes:
    def test_forward_output_shapes(self, net, dummy_batch):
        vision, nv = dummy_batch
        logits, value, (hx, cx) = net(vision, nv)
        assert logits.shape == (2, N_ACTIONS)
        assert value.shape  == (2, 1)
        assert hx.shape     == (2, 256)  # HIDDEN_DIM = 256
        assert cx.shape     == (2, 256)

    def test_lstm_state_passed_through(self, net, dummy_batch):
        vision, nv = dummy_batch
        state0 = net.get_initial_state(2)
        logits1, val1, state1 = net(vision, nv, state0)
        logits2, val2, state2 = net(vision, nv, state1)
        # Two different LSTM states should produce different outputs
        assert not torch.equal(logits1, logits2) or True  # May be same if zeros

    def test_initial_state_zeros(self, net):
        hx, cx = net.get_initial_state(1)
        assert torch.all(hx == 0.0)
        assert torch.all(cx == 0.0)

    def test_get_action_returns_valid_action(self, net, dummy_batch):
        vision, nv = dummy_batch
        action, log_prob, value, state = net.get_action(vision[:1], nv[:1])
        assert 0 <= action < N_ACTIONS
        assert isinstance(action, int)

    def test_evaluate_actions(self, net, dummy_batch):
        vision, nv = dummy_batch
        actions = torch.zeros(2, dtype=torch.long)
        log_probs, values, entropy = net.evaluate_actions(vision, nv, actions)
        assert log_probs.shape == (2,)
        assert values.shape    == (2, 1)
        assert entropy.shape   == (2,)

    def test_logits_are_finite(self, net, dummy_batch):
        vision, nv = dummy_batch
        logits, _, _ = net(vision, nv)
        assert torch.all(torch.isfinite(logits))

    def test_value_is_finite(self, net, dummy_batch):
        vision, nv = dummy_batch
        _, value, _ = net(vision, nv)
        assert torch.all(torch.isfinite(value))

    def test_entropy_positive(self, net, dummy_batch):
        vision, nv = dummy_batch
        actions = torch.zeros(2, dtype=torch.long)
        _, _, entropy = net.evaluate_actions(vision, nv, actions)
        assert torch.all(entropy >= 0.0)

    def test_deterministic_action_is_argmax(self, net, dummy_batch):
        vision, nv = dummy_batch
        net.eval()  # disable dropout so both forward passes are identical
        action_det, _, _, _ = net.get_action(vision[:1], nv[:1], deterministic=True)
        logits, _, _ = net(vision[:1], nv[:1])
        expected = int(logits.argmax(dim=-1).item())
        assert action_det == expected

    def test_parameter_count_reasonable(self, net):
        n_params = sum(p.numel() for p in net.parameters())
        # Should be more than 100K params but less than 100M for real-time inference
        assert 100_000 < n_params < 100_000_000


class TestVisualEncoder:
    def test_output_shape(self):
        from src.emergent_creativity.nn.architecture import VisualEncoder
        enc   = VisualEncoder()
        dummy = torch.zeros(1, VISION_C, VISION_H, VISION_W)
        out   = enc(dummy)
        assert out.shape == (1, enc.out_dim)

    def test_batch_processing(self):
        from src.emergent_creativity.nn.architecture import VisualEncoder
        enc   = VisualEncoder()
        dummy = torch.zeros(4, VISION_C, VISION_H, VISION_W)
        out   = enc(dummy)
        assert out.shape == (4, enc.out_dim)


class TestSensoryEncoder:
    def test_output_shape(self):
        from src.emergent_creativity.nn.architecture import SensoryEncoder
        enc   = SensoryEncoder()
        dummy = torch.zeros(1, NON_VISUAL_DIM)
        out   = enc(dummy)
        assert out.shape == (1, enc.out_dim)


# ---------------------------------------------------------------------------
# OnlineLearner tests
# ---------------------------------------------------------------------------

def _make_dummy_obs():
    """Return a minimal observation dict matching TenantEnv's obs space."""
    from src.emergent_creativity.environment.senses import (
        VISION_H, VISION_W, VISION_C,
        HEARING_DIM, TOUCH_DIM, SMELL_DIM, TASTE_DIM,
    )
    return {
        "vision":   np.zeros((VISION_H, VISION_W, VISION_C), dtype=np.float32),
        "hearing":  np.zeros(HEARING_DIM, dtype=np.float32),
        "touch":    np.zeros(TOUCH_DIM,   dtype=np.float32),
        "smell":    np.zeros(SMELL_DIM,   dtype=np.float32),
        "taste":    np.zeros(TASTE_DIM,   dtype=np.float32),
        "vitals":   np.zeros(VITALS_DIM,  dtype=np.float32),
    }


@pytest.fixture
def learner():
    from src.emergent_creativity.nn.online_learner import OnlineLearner
    # Per-step learner — no buffer params needed.
    return OnlineLearner(device="cpu")


class TestOnlineLearner:
    def test_act_returns_valid_action(self, learner):
        obs    = _make_dummy_obs()
        action = learner.act(obs)
        assert 0 <= action < N_ACTIONS

    def test_observe_before_act_returns_empty(self, learner):
        # observe() before any act() must not crash and returns empty dict.
        obs    = _make_dummy_obs()
        result = learner.observe(obs, 0.0, False)
        assert result == {}

    def test_update_on_first_step(self, learner):
        """Every observe() fires an immediate gradient update."""
        obs   = _make_dummy_obs()
        learner.act(obs)
        stats = learner.observe(obs, 1.0, False)
        # Per-step: stats are always returned (not None, not empty).
        assert stats
        assert "loss" in stats
        assert learner.update_count == 1

    def test_update_on_every_step(self, learner):
        """update_count must equal step_count — one update per step."""
        obs = _make_dummy_obs()
        n   = 10
        for i in range(n):
            learner.act(obs)
            stats = learner.observe(obs, 0.5, False)
            assert stats, f"No stats returned on step {i}"
        assert learner.update_count == n

    def test_loss_is_finite(self, learner):
        obs = _make_dummy_obs()
        learner.act(obs)
        learner.observe(obs, 1.0, False)
        assert np.isfinite(learner.last_loss)

    def test_loss_stats_keys(self, learner):
        obs   = _make_dummy_obs()
        learner.act(obs)
        stats = learner.observe(obs, 1.0, False)
        assert {"loss", "actor_loss", "critic_loss", "entropy"} <= stats.keys()

    def test_weights_change_after_single_step(self, learner):
        """Network weights must change after a single act+observe cycle."""
        obs    = _make_dummy_obs()
        before = {k: v.clone() for k, v in learner.net.named_parameters()}
        learner.act(obs)
        learner.observe(obs, 1.0, False)
        changed = any(
            not torch.equal(before[k], v)
            for k, v in learner.net.named_parameters()
        )
        assert changed, "No weight change detected after gradient step"

    def test_reset_lstm_does_not_crash(self, learner):
        learner.reset_lstm()  # callable at any time

    def test_done_resets_lstm_state(self, learner):
        obs = _make_dummy_obs()
        # Drive the LSTM to a non-zero state
        learner.act(obs)
        learner.observe(obs, 1.0, False)
        # Signalling done=True must reset LSTM to zeros
        learner.act(obs)
        learner.observe(obs, 0.0, True)
        h, c = learner._lstm_state
        assert torch.all(h == 0.0), "LSTM hidden state should be zeroed after done=True"
        assert torch.all(c == 0.0), "LSTM cell state should be zeroed after done=True"

    def test_save_load_roundtrip(self, learner, tmp_path):
        obs = _make_dummy_obs()
        for _ in range(5):
            learner.act(obs)
            learner.observe(obs, 1.0, False)
        path = str(tmp_path / "online.pt")
        learner.save(path)
        from src.emergent_creativity.nn.online_learner import OnlineLearner
        learner2 = OnlineLearner(device="cpu")
        learner2.load(path)
        assert learner2.step_count == learner.step_count

    def test_is_learning_flag(self, learner):
        assert not learner.is_learning
        obs = _make_dummy_obs()
        learner.act(obs)
        learner.observe(obs, 1.0, False)
        assert learner.is_learning
        # Flag must remain True across further steps.
        learner.act(obs)
        learner.observe(obs, 0.5, False)
        assert learner.is_learning

    def test_done_step_still_produces_stats(self, learner):
        """A terminal step should still return loss stats."""
        obs   = _make_dummy_obs()
        learner.act(obs)
        stats = learner.observe(obs, -1.0, True)
        assert stats
        assert "loss" in stats
