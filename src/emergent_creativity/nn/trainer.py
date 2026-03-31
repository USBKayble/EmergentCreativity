"""
trainer.py
==========
PPO (Proximal Policy Optimisation) training loop for the Tenant NN.

Key features
------------
* GPU acceleration via PyTorch CUDA (falls back to CPU automatically).
* TensorBoard logging.
* Checkpoint saving and loading.
* Configurable via constructor kwargs or from the environment config.

Usage
-----
::

    from emergent_creativity.nn.trainer import PPOTrainer
    from emergent_creativity.sim_env import TenantEnv

    env = TenantEnv()
    trainer = PPOTrainer(env)
    trainer.train(total_timesteps=1_000_000)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

from .architecture import TenantNetwork, _TORCH as ARCH_TORCH
from ..environment.senses import VISION_H, VISION_W, VISION_C, TOTAL_SENSORY_DIM
from ..tenant.actions import N_ACTIONS

VITALS_DIM = 4


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch required. pip install torch")


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores transitions collected over *n_steps* steps.
    Used for a single PPO update cycle.
    """

    def __init__(
        self,
        n_steps: int,
        vision_shape: Tuple[int, int, int],
        non_visual_dim: int,
        device: "torch.device",
    ) -> None:
        _require_torch()
        H, W, C = vision_shape
        self.n_steps  = n_steps
        self.device   = device
        self.ptr      = 0
        self.full     = False

        self.visions     = torch.zeros(n_steps, C, H, W,          device=device)
        self.non_visuals = torch.zeros(n_steps, non_visual_dim,   device=device)
        self.actions     = torch.zeros(n_steps,                   device=device, dtype=torch.long)
        self.log_probs   = torch.zeros(n_steps,                   device=device)
        self.rewards     = torch.zeros(n_steps,                   device=device)
        self.values      = torch.zeros(n_steps,                   device=device)
        self.dones       = torch.zeros(n_steps,                   device=device)
        self.advantages  = torch.zeros(n_steps,                   device=device)
        self.returns     = torch.zeros(n_steps,                   device=device)

    def add(
        self,
        vision: "torch.Tensor",
        non_visual: "torch.Tensor",
        action: int,
        log_prob: "torch.Tensor",
        reward: float,
        value: "torch.Tensor",
        done: bool,
    ) -> None:
        i = self.ptr
        self.visions[i]     = vision
        self.non_visuals[i] = non_visual
        self.actions[i]     = action
        self.log_probs[i]   = log_prob.detach()
        self.rewards[i]     = reward
        self.values[i]      = value.detach().squeeze()
        self.dones[i]       = float(done)
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True
            self.ptr  = 0

    def compute_returns_advantages(
        self,
        last_value: "torch.Tensor",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Generalised Advantage Estimation (GAE)."""
        last_value = last_value.detach().squeeze()
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val   = last_value
                next_done  = self.dones[t]
            else:
                next_val   = self.values[t + 1]
                next_done  = self.dones[t + 1]
            delta  = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            gae    = delta + gamma * gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield mini-batches of (vision, non_visual, action, log_prob, adv, ret)."""
        indices = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                self.visions[idx],
                self.non_visuals[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Proximal Policy Optimisation trainer.

    Parameters
    ----------
    env               : gym.Env – the TenantEnv instance
    learning_rate     : float
    n_steps           : int   – rollout steps per update
    batch_size        : int   – mini-batch size
    n_epochs          : int   – update epochs per rollout
    gamma             : float – discount factor
    gae_lambda        : float – GAE lambda
    clip_range        : float – PPO clip epsilon
    ent_coef          : float – entropy bonus coefficient
    vf_coef           : float – value function loss coefficient
    max_grad_norm     : float – gradient clipping
    log_dir           : str   – TensorBoard log directory
    save_dir          : str   – checkpoint directory
    save_freq         : int   – save checkpoint every N steps
    device            : str   – "cuda", "cpu", or "auto"
    """

    def __init__(
        self,
        env,
        learning_rate:  float = 3e-4,
        n_steps:        int   = 2048,
        batch_size:     int   = 64,
        n_epochs:       int   = 10,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_range:     float = 0.2,
        ent_coef:       float = 0.01,
        vf_coef:        float = 0.5,
        max_grad_norm:  float = 0.5,
        log_dir:        str   = "logs/tensorboard",
        save_dir:       str   = "checkpoints",
        save_freq:      int   = 10_000,
        device:         str   = "auto",
    ) -> None:
        _require_torch()
        self.env = env

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Trainer] Using device: {self.device}")

        self.net = TenantNetwork(n_actions=N_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-5)

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

        non_visual_dim = TOTAL_SENSORY_DIM + VITALS_DIM
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            vision_shape=(VISION_H, VISION_W, VISION_C),
            non_visual_dim=non_visual_dim,
            device=self.device,
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        if _TB:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._lstm_state: Optional[Tuple] = None
        self._global_step = 0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 1_000_000) -> None:
        """Run PPO training for *total_timesteps* environment steps."""
        obs, _ = self.env.reset()
        self._lstm_state = self.net.get_initial_state(1, device=self.device)
        ep_reward = 0.0
        ep_len    = 0
        ep_count  = 0
        start_t   = time.time()

        while self._global_step < total_timesteps:
            # ---- Rollout collection ----
            for _ in range(self.n_steps):
                vision_t, non_vis_t = self._obs_to_tensors(obs)
                action, log_prob, value, new_state = self.net.get_action(
                    vision_t, non_vis_t, self._lstm_state
                )
                self._lstm_state = new_state

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(
                    vision_t.squeeze(0),
                    non_vis_t.squeeze(0),
                    action,
                    log_prob,
                    reward,
                    value,
                    done,
                )

                ep_reward += reward
                ep_len    += 1
                self._global_step += 1

                if done:
                    ep_count += 1
                    if self._writer:
                        self._writer.add_scalar("episode/reward",  ep_reward, self._global_step)
                        self._writer.add_scalar("episode/length",  ep_len,    self._global_step)
                    ep_reward = 0.0
                    ep_len    = 0
                    obs, _ = self.env.reset()
                    self._lstm_state = self.net.get_initial_state(1, device=self.device)
                else:
                    obs = next_obs

                if self._global_step % self.save_freq == 0:
                    self.save(f"checkpoint_{self._global_step}.pt")

            # ---- PPO update ----
            with torch.no_grad():
                vision_t, non_vis_t = self._obs_to_tensors(obs)
                _, last_value, _ = self.net(vision_t, non_vis_t, self._lstm_state)
            self.buffer.compute_returns_advantages(last_value, self.gamma, self.gae_lambda)
            update_info = self._update()

            if self._writer:
                for k, v in update_info.items():
                    self._writer.add_scalar(f"train/{k}", v, self._global_step)

            elapsed = time.time() - start_t
            fps = self._global_step / max(elapsed, 1e-9)
            print(
                f"Step {self._global_step:>8d}/{total_timesteps}  "
                f"Episodes: {ep_count}  "
                f"FPS: {fps:.0f}  "
                f"Loss: {update_info.get('loss', 0.0):.4f}"
            )

        if self._writer:
            self._writer.close()
        print("[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # PPO update step
    # ------------------------------------------------------------------

    def _update(self) -> dict:
        norm_adv = self.buffer.advantages
        adv_mean = norm_adv.mean()
        adv_std  = norm_adv.std() + 1e-8
        norm_adv = (norm_adv - adv_mean) / adv_std

        all_losses = []
        all_pg     = []
        all_vf     = []
        all_ent    = []
        all_clip   = []

        for _ in range(self.n_epochs):
            for vis_b, nv_b, act_b, oldlp_b, adv_b, ret_b in self.buffer.get_batches(self.batch_size):
                # Normalise advantages for this mini-batch
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                log_probs, values, entropy = self.net.evaluate_actions(
                    vis_b, nv_b, act_b
                )

                ratio = torch.exp(log_probs - oldlp_b)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()

                values = values.squeeze()
                vf_loss = F.mse_loss(values, ret_b)

                ent_loss = entropy.mean()
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_losses.append(loss.item())
                all_pg.append(pg_loss.item())
                all_vf.append(vf_loss.item())
                all_ent.append(ent_loss.item())
                all_clip.append(clip_frac.item())

        return {
            "loss":      float(np.mean(all_losses)),
            "pg_loss":   float(np.mean(all_pg)),
            "vf_loss":   float(np.mean(all_vf)),
            "entropy":   float(np.mean(all_ent)),
            "clip_frac": float(np.mean(all_clip)),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_tensors(
        self, obs: dict
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        vision   = torch.from_numpy(obs["vision"]).float()        # (H, W, C)
        vision   = vision.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, C, H, W)
        non_vis  = np.concatenate([
            obs["hearing"], obs["touch"], obs["smell"],
            obs["taste"],  obs["vitals"],
        ])
        non_vis  = torch.from_numpy(non_vis).float().unsqueeze(0).to(self.device)
        return vision, non_vis

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save({
            "step":       self._global_step,
            "model":      self.net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)
        print(f"[Trainer] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self._global_step = ck.get("step", 0)
        print(f"[Trainer] Loaded checkpoint from {path} (step {self._global_step})")
