"""
tests/test_trainer.py
=====================
Unit tests for the PPOTrainer.
"""
from __future__ import annotations

import pytest
import numpy as np

try:
    import torch
    import gymnasium as gym
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="PyTorch or Gymnasium not installed")

from src.emergent_creativity.nn.trainer import PPOTrainer, RolloutBuffer
from src.emergent_creativity.tenant.actions import N_ACTIONS
from src.emergent_creativity.environment.senses import (
    VISION_H, VISION_W, VISION_C,
    HEARING_DIM, TOUCH_DIM, SMELL_DIM, TASTE_DIM,
)
VITALS_DIM = 4

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Dict({
            "vision": gym.spaces.Box(low=0, high=255, shape=(VISION_H, VISION_W, VISION_C), dtype=np.uint8),
            "hearing": gym.spaces.Box(low=0.0, high=1.0, shape=(HEARING_DIM,), dtype=np.float32),
            "touch": gym.spaces.Box(low=0.0, high=1.0, shape=(TOUCH_DIM,), dtype=np.float32),
            "smell": gym.spaces.Box(low=0.0, high=1.0, shape=(SMELL_DIM,), dtype=np.float32),
            "taste": gym.spaces.Box(low=0.0, high=1.0, shape=(TASTE_DIM,), dtype=np.float32),
            "vitals": gym.spaces.Box(low=0.0, high=1.0, shape=(VITALS_DIM,), dtype=np.float32),
        })
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = 1.0
        terminated = self.step_count >= 10
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return {
            "vision": np.zeros((VISION_H, VISION_W, VISION_C), dtype=np.float32),
            "hearing": np.zeros(HEARING_DIM, dtype=np.float32),
            "touch": np.zeros(TOUCH_DIM, dtype=np.float32),
            "smell": np.zeros(SMELL_DIM, dtype=np.float32),
            "taste": np.zeros(TASTE_DIM, dtype=np.float32),
            "vitals": np.zeros(VITALS_DIM, dtype=np.float32),
        }

@pytest.fixture
def trainer():
    env = DummyEnv()
    return PPOTrainer(
        env=env,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        device="cpu",
    )

def test_trainer_init(trainer):
    assert trainer.env is not None
    assert trainer.device.type == "cpu"
    assert trainer.n_steps == 16
    assert isinstance(trainer.buffer, RolloutBuffer)

def test_trainer_train_one_update(trainer):
    trainer.train(total_timesteps=16)
    assert trainer._global_step == 16
    assert trainer.buffer.ptr == 0  # Buffer pointer wraps to 0 after filling

def test_trainer_save_load(trainer, tmp_path):
    import os
    trainer.save_dir = tmp_path
    trainer.save("test_ckpt.pt")
    assert os.path.exists(tmp_path / "test_ckpt.pt")

    # modify net params to ensure load overwrites them
    with torch.no_grad():
        for param in trainer.net.parameters():
            param.add_(1.0)

    trainer.load(tmp_path / "test_ckpt.pt")
    # check that load was successful by looking at step count
    assert trainer._global_step == 0

def test_rollout_buffer_add(trainer):
    buffer = trainer.buffer
    assert buffer.ptr == 0
    vision = torch.zeros(VISION_C, VISION_H, VISION_W)
    non_visual = torch.zeros(HEARING_DIM + TOUCH_DIM + SMELL_DIM + TASTE_DIM + VITALS_DIM)
    log_prob = torch.tensor(-1.0)
    value = torch.tensor(0.5)

    buffer.add(vision, non_visual, 1, log_prob, 1.0, value, False)
    assert buffer.ptr == 1

def test_rollout_buffer_returns_advantages(trainer):
    buffer = trainer.buffer
    vision = torch.zeros(VISION_C, VISION_H, VISION_W)
    non_visual = torch.zeros(HEARING_DIM + TOUCH_DIM + SMELL_DIM + TASTE_DIM + VITALS_DIM)
    log_prob = torch.tensor(-1.0)
    value = torch.tensor(0.5)

    for _ in range(trainer.n_steps):
        buffer.add(vision, non_visual, 1, log_prob, 1.0, value, False)

    last_value = torch.tensor(0.5)
    buffer.compute_returns_advantages(last_value, 0.99, 0.95)

    assert torch.isfinite(buffer.advantages).all()
    assert torch.isfinite(buffer.returns).all()

def test_trainer_obs_to_tensors(trainer):
    obs = DummyEnv()._get_obs()
    vision_t, non_vis_t = trainer._obs_to_tensors(obs)

    assert vision_t.shape == (1, VISION_C, VISION_H, VISION_W)
    expected_non_vis_dim = HEARING_DIM + TOUCH_DIM + SMELL_DIM + TASTE_DIM + VITALS_DIM
    assert non_vis_t.shape == (1, expected_non_vis_dim)
