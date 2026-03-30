"""
online_learner.py
=================
Truly simultaneous inference + learning agent.

The network is permanently in ``train()`` mode and its weights update after
**every single** environment step.  There is no conceptual separation between
"data collection" and "gradient updates" — ``act()`` and ``observe()`` together
form one complete forward+backward training step on each interaction.

Algorithm: Online Actor-Critic with 1-step TD(0) advantage.

1. ``act(obs)``  – forward pass (WITH gradient tracking) → action.
   Stores the value estimate and log-probability tensors for the backward pass.
2. ``observe(next_obs, reward, done)``  – bootstraps the TD target from
   *next_obs*, computes the Actor-Critic loss, and **immediately** steps the
   optimiser.  This happens on every single call — there is no buffer to fill.

Because the gradient step fires on every ``observe()`` call the policy
improves continuously while the environment runs.  The whole system *is*
a training loop that outputs actions as it trains.

Usage
-----
::

    learner          = OnlineLearner()
    obs, _           = env.reset()

    while running:
        action              = learner.act(obs)
        next_obs, r, te, tr, _ = env.step(action)
        done                = te or tr
        learner.observe(next_obs, r, done)   # ← gradient step happens here
        obs = next_obs
        if done:
            obs, _ = env.reset()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    _TORCH = True
except ImportError:
    _TORCH = False

from .architecture import TenantNetwork
from ..environment.senses import VISION_H, VISION_W, VISION_C, TOTAL_SENSORY_DIM
from ..tenant.actions import N_ACTIONS

VITALS_DIM = 4
NON_VISUAL_DIM = TOTAL_SENSORY_DIM + VITALS_DIM


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch is required for OnlineLearner. pip install torch")


class OnlineLearner:
    """
    Per-step online Actor-Critic: the network learns on **every single**
    environment interaction.

    Inference and training are unified — calling ``act(obs)`` followed by
    ``observe(next_obs, reward, done)`` is one complete forward + backward pass.
    The gradient step fires unconditionally on every ``observe()`` call; there
    is no rollout buffer to fill and no separate update phase.

    Parameters
    ----------
    learning_rate  : Adam learning rate (default 3e-4).
    gamma          : discount factor (default 0.99).
    ent_coef       : entropy bonus coefficient (default 0.01).
    vf_coef        : value loss coefficient (default 0.5).
    max_grad_norm  : gradient clipping (default 0.5).
    device         : "auto", "cuda", or "cpu".
    save_dir       : directory for automatic checkpoints.
    save_freq      : save a checkpoint every N steps; 0 = disabled.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        save_dir: str = "checkpoints",
        save_freq: int = 0,
        use_amp: bool = True,
    ) -> None:
        _require_torch()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.use_amp = use_amp and self.device.type == "cuda"

        self.net = TenantNetwork(n_actions=N_ACTIONS).to(self.device)
        self.net.train()

        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-5)

        self._lstm_state: Optional[Tuple] = self.net.get_initial_state(
            1, device=self.device
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self._gradient_accumulation_steps = 4
        self._accumulated_steps = 0

        self._pending_value = None
        self._pending_log_prob = None
        self._pending_entropy = None

        # Statistics
        self._step_count: int = 0
        self._last_loss: float = 0.0
        self._last_stats: Dict = {}

        print(
            f"[OnlineLearner] Initialised on {self.device}. "
            f"Gradient update on every environment step."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, obs: dict) -> int:
        """
        Forward pass → action.

        The computational graph is retained so that the value estimate and
        log-probability computed here can be used for the backward pass in
        the subsequent :meth:`observe` call.

        The LSTM hidden state is propagated but detached from the graph after
        each step (truncated BPTT with window = 1) to prevent unbounded memory
        growth while still preserving temporal context.

        Must be followed immediately by :meth:`observe`.
        """
        vision_t, non_vis_t = self._obs_to_tensors(obs)

        # Forward pass — keep computational graph (no torch.no_grad).
        logits, value, new_state = self.net(vision_t, non_vis_t, self._lstm_state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        # Store tensors needed for the backward pass in observe().
        self._pending_log_prob = dist.log_prob(action)  # (1,)  grad retained
        self._pending_value = value  # (1,1) grad retained
        self._pending_entropy = dist.entropy()  # (1,)

        # Detach the LSTM state: gradients must not flow through the full
        # episode history (truncated BPTT).
        self._lstm_state = (new_state[0].detach(), new_state[1].detach())

        return int(action.item())

    def observe(self, next_obs: dict, reward: float, done: bool) -> Dict:
        """
        Complete the transition and **immediately** update the network weights.

        This is called after every ``env.step()`` and always performs a full
        gradient step — there is no buffering or waiting.

        The 1-step TD(0) target is bootstrapped from *next_obs* (or set to
        *reward* alone when the episode ends).

        Parameters
        ----------
        next_obs : dict  – observation returned by ``env.step()``.
        reward   : float – scalar reward from ``env.step()``.
        done     : bool  – True if the episode terminated or was truncated.

        Returns
        -------
        dict with keys ``loss``, ``actor_loss``, ``critic_loss``, ``entropy``.
        Returns ``{}`` if :meth:`act` was not called first.
        """
        if self._pending_value is None:
            return {}

        # ---- 1-step TD target ----------------------------------------
        if done:
            # Terminal state: future value is zero.
            td_target = torch.tensor(
                [[reward]], device=self.device, dtype=torch.float32
            )
            # Reset LSTM for the next episode.
            self._lstm_state = self.net.get_initial_state(1, device=self.device)
        else:
            next_vision, next_nv = self._obs_to_tensors(next_obs)
            with torch.no_grad():
                _, next_value, _ = self.net(next_vision, next_nv, self._lstm_state)
            td_target = reward + self.gamma * next_value  # (1, 1)

        # ---- Actor-Critic loss ----------------------------------------
        # Advantage: stop gradient so only the actor head is guided by it.
        advantage = (td_target - self._pending_value).detach()
        actor_loss = -(self._pending_log_prob * advantage.squeeze()).mean()
        critic_loss = F.mse_loss(self._pending_value, td_target)
        entropy = self._pending_entropy.mean()

        loss = (
            actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
        ) / self._gradient_accumulation_steps

        # ---- Gradient step --------------------------------------------
        self.optimizer.zero_grad()
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accumulated_steps += 1

        if self._accumulated_steps >= self._gradient_accumulation_steps:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self._accumulated_steps = 0

        # Free the retained graph tensors.
        self._pending_log_prob = None
        self._pending_value = None
        self._pending_entropy = None

        # ---- Statistics -----------------------------------------------
        self._step_count += 1
        stats = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }
        self._last_loss = stats["loss"]
        self._last_stats = stats

        if self.save_freq > 0 and self._step_count % self.save_freq == 0:
            self._auto_save()

        return stats

    def reset_lstm(self) -> None:
        """Manually reset the LSTM hidden state (e.g. on a forced episode reset)."""
        self._lstm_state = self.net.get_initial_state(1, device=self.device)

    # ------------------------------------------------------------------
    # Properties (for UI / external monitoring)
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Total gradient updates performed (one per completed act+observe cycle)."""
        return self._step_count

    @property
    def update_count(self) -> int:
        """Alias for :attr:`step_count` — every step is an update."""
        return self._step_count

    @property
    def last_loss(self) -> float:
        return self._last_loss

    @property
    def last_stats(self) -> Dict:
        return self._last_stats

    @property
    def is_learning(self) -> bool:
        """True once the network has performed at least one gradient update
        (remains True thereafter)."""
        return self._step_count > 0

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_to_tensors(self, obs: dict) -> Tuple["torch.Tensor", "torch.Tensor"]:
        vision = torch.from_numpy(obs["vision"]).float()
        vision = vision.permute(2, 0, 1).unsqueeze(0).to(self.device)
        non_vis = np.concatenate(
            [
                obs["hearing"],
                obs["touch"],
                obs["smell"],
                obs["taste"],
                obs["vitals"],
            ]
        )
        non_vis = torch.from_numpy(non_vis).float().unsqueeze(0).to(self.device)
        return vision, non_vis

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model + optimiser state to *path*."""
        torch.save(
            {
                "step": self._step_count,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[OnlineLearner] Saved → {path}")

    def load(self, path: str) -> None:
        """Load model + optimiser state from *path*."""
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self._step_count = ck.get("step", 0)
        print(f"[OnlineLearner] Loaded {path}  (step {self._step_count})")

    def _auto_save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save(str(self.save_dir / f"online_{self._step_count}.pt"))
