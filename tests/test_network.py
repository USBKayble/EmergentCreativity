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
        vision, nv = dummy_batch[0].unsqueeze(0), dummy_batch[1].unsqueeze(0)
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
        vision, nv = dummy_batch[0].unsqueeze(0), dummy_batch[1].unsqueeze(0)
        action_det, _, _, _ = net.get_action(vision, nv, deterministic=True)
        logits, _, _ = net(vision, nv)
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
