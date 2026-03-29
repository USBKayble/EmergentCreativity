"""
online_learner.py
=================
Online PPO agent that **learns while inferring** — weights are updated
in-place every *n_steps* environment steps, without any separate training
loop.

How it works
------------
1. ``act(obs)`` – performs a forward pass (inference) and records the
   partial transition *(obs, action, log_prob, value)*.
2. ``observe(reward, done)`` – completes the transition with the reward
   and terminal flag.  When the rolling buffer is full it triggers a
   compact PPO update cycle (few epochs, small mini-batches) on the GPU
   (or CPU if unavailable), then clears the buffer and resumes.

Because the update happens inside the simulation loop the agent adapts
its policy in real time, every few seconds of wall-clock time.

Usage (inside the viewer loop)
-------------------------------
::

    learner = OnlineLearner()
    obs, _ = env.reset()

    while running:
        action = learner.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        learner.observe(reward, terminated or truncated)

        if terminated or truncated:
            obs, _ = env.reset()
            learner.reset_lstm()
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
from .trainer import RolloutBuffer
from ..environment.senses import VISION_H, VISION_W, VISION_C, TOTAL_SENSORY_DIM
from ..tenant.actions import N_ACTIONS

VITALS_DIM = 4
NON_VISUAL_DIM = TOTAL_SENSORY_DIM + VITALS_DIM


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch is required for OnlineLearner. pip install torch")


class OnlineLearner:
    """
    Continual online PPO agent: infers *and* learns in the same loop.

    Parameters
    ----------
    n_steps        : rollout length before each weight update (default 128).
                     Smaller → more frequent updates; larger → more stable
                     gradient estimates.  128 at 30 FPS ≈ update every 4 s.
    batch_size     : mini-batch size for PPO update (default 32).
    n_epochs       : PPO epochs per rollout (default 4).
    learning_rate  : Adam learning rate (default 3e-4).
    gamma          : discount factor (default 0.99).
    gae_lambda     : GAE lambda (default 0.95).
    clip_range     : PPO clip epsilon (default 0.2).
    ent_coef       : entropy bonus (default 0.01).
    vf_coef        : value loss coefficient (default 0.5).
    max_grad_norm  : gradient clipping (default 0.5).
    device         : "auto", "cuda", or "cpu".
    save_dir       : directory for automatic checkpoints.
    save_freq      : save a checkpoint every N steps (0 = disabled).
    """

    def __init__(
        self,
        n_steps:        int   = 128,
        batch_size:     int   = 32,
        n_epochs:       int   = 4,
        learning_rate:  float = 3e-4,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_range:     float = 0.2,
        ent_coef:       float = 0.01,
        vf_coef:        float = 0.5,
        max_grad_norm:  float = 0.5,
        device:         str   = "auto",
        save_dir:       str   = "checkpoints",
        save_freq:      int   = 0,
    ) -> None:
        _require_torch()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_steps       = n_steps
        self.batch_size    = batch_size
        self.n_epochs      = n_epochs
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_range    = clip_range
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.save_freq     = save_freq
        self.save_dir      = Path(save_dir)

        # Network and optimiser
        self.net = TenantNetwork(n_actions=N_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=learning_rate, eps=1e-5
        )

        # Rolling rollout buffer (small, for frequent online updates)
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            vision_shape=(VISION_H, VISION_W, VISION_C),
            non_visual_dim=NON_VISUAL_DIM,
            device=self.device,
        )

        # LSTM hidden state (persistent across steps / episodes)
        self._lstm_state: Optional[Tuple] = self.net.get_initial_state(
            1, device=self.device
        )

        # Pending transition fields (filled by act(), completed by observe())
        self._pending_vision:     Optional["torch.Tensor"] = None
        self._pending_non_visual: Optional["torch.Tensor"] = None
        self._pending_action:     Optional[int]            = None
        self._pending_log_prob:   Optional["torch.Tensor"] = None
        self._pending_value:      Optional["torch.Tensor"] = None

        # Statistics
        self._step_count:   int   = 0
        self._update_count: int   = 0
        self._last_loss:    float = 0.0
        self._last_stats:   Dict  = {}

        print(
            f"[OnlineLearner] Initialised on {self.device}. "
            f"Update every {n_steps} steps."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, obs: dict) -> int:
        """
        Compute and return the next action.

        Performs a forward pass through the network (in *training* mode so
        batch-norm / dropout behave correctly during the online updates),
        records the partial transition, and returns the chosen action.

        Must be followed by a call to :meth:`observe` with the reward and
        terminal flag received from the environment.
        """
        vision_t, non_vis_t = self._obs_to_tensors(obs)

        self.net.train()  # train mode so batch-norm/dropout behave correctly during online updates
        with torch.no_grad():
            action, log_prob, value, new_state = self.net.get_action(
                vision_t, non_vis_t, self._lstm_state
            )
        self._lstm_state = new_state

        # Store partial transition
        self._pending_vision     = vision_t.squeeze(0)
        self._pending_non_visual = non_vis_t.squeeze(0)
        self._pending_action     = action
        self._pending_log_prob   = log_prob
        self._pending_value      = value

        return action

    def observe(self, reward: float, done: bool) -> Optional[Dict]:
        """
        Complete the pending transition with *reward* and *done*.

        If the rollout buffer is now full, runs a PPO update cycle and
        returns a dict with loss statistics; otherwise returns ``None``.

        Parameters
        ----------
        reward : float  – reward received from env.step()
        done   : bool   – True if the episode ended (terminated or truncated)
        """
        if self._pending_action is None:
            return None  # act() was never called — nothing to record

        self.buffer.add(
            self._pending_vision,
            self._pending_non_visual,
            self._pending_action,
            self._pending_log_prob,
            reward,
            self._pending_value,
            done,
        )

        self._step_count += 1

        # Clear pending transition
        self._pending_action = None

        if done:
            # Reset LSTM state at episode boundary
            self._lstm_state = self.net.get_initial_state(1, device=self.device)

        # Trigger update when buffer is full
        if self.buffer.full:
            stats = self._update()
            self._last_stats  = stats
            self._last_loss   = stats.get("loss", 0.0)
            self._update_count += 1
            # Save based on update count (not step count) so the frequency
            # is independent of n_steps vs save_freq alignment.
            if self.save_freq > 0 and self._update_count % self.save_freq == 0:
                self._auto_save()
            return stats

        return None

    def reset_lstm(self) -> None:
        """Manually reset the LSTM hidden state (call on episode reset)."""
        self._lstm_state = self.net.get_initial_state(1, device=self.device)

    # ------------------------------------------------------------------
    # Properties for UI display
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def last_loss(self) -> float:
        return self._last_loss

    @property
    def last_stats(self) -> Dict:
        return self._last_stats

    @property
    def is_learning(self) -> bool:
        """True once the network has performed at least one PPO update.
        This flag is permanently True after the first update (never resets)."""
        return self._update_count > 0

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update(self) -> Dict:
        """Run PPO update on the filled rollout buffer."""
        # Bootstrap last value for GAE
        with torch.no_grad():
            if self._pending_vision is not None:
                # There is a fresh pending observation — use it
                vis_t = self._pending_vision.unsqueeze(0)
                nv_t  = self._pending_non_visual.unsqueeze(0)
            else:
                # Fall back to zeros (episode ended with done=True)
                vis_t = torch.zeros(
                    1, VISION_C, VISION_H, VISION_W, device=self.device
                )
                nv_t  = torch.zeros(1, NON_VISUAL_DIM, device=self.device)
            _, last_value, _ = self.net(vis_t, nv_t, self._lstm_state)

        self.buffer.compute_returns_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        norm_adv = self.buffer.advantages
        adv_mean = norm_adv.mean()
        adv_std  = norm_adv.std() + 1e-8
        norm_adv = (norm_adv - adv_mean) / adv_std

        all_losses, all_pg, all_vf, all_ent, all_clip = [], [], [], [], []

        self.net.train()
        for _ in range(self.n_epochs):
            for vis_b, nv_b, act_b, oldlp_b, adv_b, ret_b in self.buffer.get_batches(
                self.batch_size
            ):
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                log_probs, values, entropy = self.net.evaluate_actions(
                    vis_b, nv_b, act_b
                )

                ratio    = torch.exp(log_probs - oldlp_b)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss   = torch.max(pg_loss1, pg_loss2).mean()
                clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()

                vf_loss  = F.mse_loss(values.squeeze(), ret_b)
                ent_loss = entropy.mean()
                loss     = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                all_losses.append(loss.item())
                all_pg.append(pg_loss.item())
                all_vf.append(vf_loss.item())
                all_ent.append(ent_loss.item())
                all_clip.append(clip_frac.item())

        # Reset buffer for next rollout
        self.buffer.ptr  = 0
        self.buffer.full = False

        return {
            "loss":      float(np.mean(all_losses)),
            "pg_loss":   float(np.mean(all_pg)),
            "vf_loss":   float(np.mean(all_vf)),
            "entropy":   float(np.mean(all_ent)),
            "clip_frac": float(np.mean(all_clip)),
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_to_tensors(
        self, obs: dict
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        vision  = torch.from_numpy(obs["vision"]).float()
        vision  = vision.permute(2, 0, 1).unsqueeze(0).to(self.device)
        non_vis = np.concatenate([
            obs["hearing"], obs["touch"], obs["smell"],
            obs["taste"],  obs["vitals"],
        ])
        non_vis = torch.from_numpy(non_vis).float().unsqueeze(0).to(self.device)
        return vision, non_vis

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model + optimiser state to *path*."""
        torch.save(
            {
                "step":      self._step_count,
                "updates":   self._update_count,
                "model":     self.net.state_dict(),
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
        self._step_count   = ck.get("step",    0)
        self._update_count = ck.get("updates", 0)
        print(
            f"[OnlineLearner] Loaded {path}  "
            f"(step {self._step_count}, updates {self._update_count})"
        )

    def _auto_save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save(str(self.save_dir / f"online_{self._step_count}.pt"))
