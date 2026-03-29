"""
architecture.py
===============
Neuroplasticity-inspired multi-modal neural network for the tenant RL agent.

Design principles
-----------------
1. **Multi-modal input processing** – separate encoders for each sense.
2. **Temporal memory** – an LSTM cell for temporal context, enabling the
   agent to remember past experiences (essential for strategy).
3. **Neuroplasticity** – achieved via:
   - *Attention gates* that modulate feature importance dynamically,
     similar to neuromodulatory gating in biological brains.
   - *Hebbian-like learning signal* injected as an auxiliary loss to
     strengthen connections that are consistently co-activated.
   - *Dropout* encourages sparse activation (analogous to pruning).
4. **Actor-Critic outputs** for PPO:
   - Actor head  → logits over discrete action space
   - Critic head → scalar value estimate

Architecture diagram
--------------------
                  ┌──────────────────┐
  Vision (H×W×3) │   CNN Encoder    │──┐
                  └──────────────────┘  │
  Hearing (8,)   ─────────────────────►│  Fusion
  Touch   (4,)   ─────────────────────►│  + Attention ──► LSTM ──► Actor / Critic
  Smell   (SMELL_DIM,) ───────────────►│  Gate
  Taste   (TASTE_DIM,)────────────────►│
  Vitals  (4,)   ─────────────────────►│
                  └──────────────────┘
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    _TORCH = False

from ..environment.senses import (
    VISION_H, VISION_W, VISION_C,
    HEARING_DIM, TOUCH_DIM, SMELL_DIM, TASTE_DIM,
)
from ..tenant.actions import N_ACTIONS

VITALS_DIM  = 4       # hunger, energy, bladder, happiness
HIDDEN_DIM  = 256     # LSTM hidden size
CNN_OUT_DIM = 256     # flattened CNN output


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError(
            "PyTorch is required for the neural network. "
            "Install it with: pip install torch"
        )


# ---------------------------------------------------------------------------
# CNN visual encoder
# ---------------------------------------------------------------------------

if _TORCH:
    class VisualEncoder(nn.Module):
        """
        Small convolutional network that encodes (C, H, W) RGB images into a
        fixed-size feature vector.

        Architecture:
          Conv(3→32, k=8, s=4) → Conv(32→64, k=4, s=2) → Conv(64→64, k=3, s=1)
          → Flatten → Linear(→256) → ReLU
        """

        def __init__(self, in_channels: int = VISION_C, out_dim: int = CNN_OUT_DIM) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
            )
            # Compute flattened size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, VISION_H, VISION_W)
                flat_size = self.conv(dummy).numel()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, out_dim),
                nn.ReLU(inplace=True),
            )
            self.out_dim = out_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.fc(self.conv(x))


    # ---------------------------------------------------------------------------
    # Non-visual sense encoder
    # ---------------------------------------------------------------------------

    class SensoryEncoder(nn.Module):
        """
        MLP that encodes all non-visual senses + vitals into a feature vector.

        Input: [hearing | touch | smell | taste | vitals]
        """

        def __init__(self, out_dim: int = 64) -> None:
            in_dim = HEARING_DIM + TOUCH_DIM + SMELL_DIM + TASTE_DIM + VITALS_DIM
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_dim),
                nn.ReLU(inplace=True),
            )
            self.out_dim = out_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    # ---------------------------------------------------------------------------
    # Attention gate (neuroplasticity module)
    # ---------------------------------------------------------------------------

    class AttentionGate(nn.Module):
        """
        Learns to weight each feature dimension dynamically.

        This is inspired by neuromodulatory gating – dopamine / acetylcholine
        signals that upregulate attention to salient features.
        """

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x * self.gate(x)


    # ---------------------------------------------------------------------------
    # Main TenantNetwork (Actor-Critic with LSTM)
    # ---------------------------------------------------------------------------

    class TenantNetwork(nn.Module):
        """
        Multi-modal Actor-Critic network with LSTM temporal memory.

        Parameters
        ----------
        n_actions : int
        lstm_hidden : int
        """

        def __init__(
            self,
            n_actions: int = N_ACTIONS,
            lstm_hidden: int = HIDDEN_DIM,
        ) -> None:
            super().__init__()
            _require_torch()

            self.n_actions = n_actions

            # --- Encoders ---
            self.visual_enc   = VisualEncoder()
            self.sensory_enc  = SensoryEncoder(out_dim=64)

            # Fusion layer
            fusion_in = self.visual_enc.out_dim + self.sensory_enc.out_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, lstm_hidden),
                nn.LayerNorm(lstm_hidden),
                nn.ReLU(inplace=True),
            )

            # Neuroplasticity: attention gate on fused features
            self.attention = AttentionGate(lstm_hidden)

            # Temporal memory (LSTM)
            self.lstm = nn.LSTMCell(lstm_hidden, lstm_hidden)

            # Dropout for sparse activations (pruning analogue)
            self.dropout = nn.Dropout(p=0.1)

            # --- Output heads ---
            # Actor: produces action logits
            self.actor = nn.Sequential(
                nn.Linear(lstm_hidden, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_actions),
            )
            # Critic: produces value estimate
            self.critic = nn.Sequential(
                nn.Linear(lstm_hidden, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )

            self._lstm_hidden = lstm_hidden
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            # Actor output head: smaller init for better exploration
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            # Critic output: slightly larger
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

        def get_initial_state(
            self, batch_size: int = 1, device: Optional["torch.device"] = None
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Return zero LSTM hidden state (hx, cx)."""
            device = device or torch.device("cpu")
            hx = torch.zeros(batch_size, self._lstm_hidden, device=device)
            cx = torch.zeros(batch_size, self._lstm_hidden, device=device)
            return hx, cx

        def forward(
            self,
            vision: "torch.Tensor",
            non_visual: "torch.Tensor",
            lstm_state: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
            """
            Parameters
            ----------
            vision      : (B, C, H, W) float32
            non_visual  : (B, hearing+touch+smell+taste+vitals) float32
            lstm_state  : (hx, cx) from previous step, or None for zeros

            Returns
            -------
            action_logits : (B, n_actions)
            value         : (B, 1)
            new_lstm_state: (hx, cx)
            """
            B = vision.shape[0]

            if lstm_state is None:
                lstm_state = self.get_initial_state(B, device=vision.device)

            # Encode each modality
            vis_feat    = self.visual_enc(vision)           # (B, 256)
            sense_feat  = self.sensory_enc(non_visual)      # (B, 64)

            # Fuse
            fused = self.fusion(torch.cat([vis_feat, sense_feat], dim=1))  # (B, 256)

            # Neuroplastic attention gate
            attended = self.attention(fused)                # (B, 256)

            # LSTM temporal integration
            hx, cx = self.lstm(attended, lstm_state)        # (B, 256)
            hx = self.dropout(hx)

            # Actor / Critic
            action_logits = self.actor(hx)                  # (B, n_actions)
            value         = self.critic(hx)                 # (B, 1)

            return action_logits, value, (hx, cx)

        def get_action(
            self,
            vision: "torch.Tensor",
            non_visual: "torch.Tensor",
            lstm_state: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
            deterministic: bool = False,
        ) -> Tuple[int, "torch.Tensor", "torch.Tensor",
                   Tuple["torch.Tensor", "torch.Tensor"]]:
            """
            Sample (or take best) action.

            Returns (action_int, log_prob, value, new_lstm_state)
            """
            with torch.no_grad():
                logits, value, new_state = self.forward(vision, non_visual, lstm_state)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
            return int(action.item()), log_prob, value, new_state

        def evaluate_actions(
            self,
            vision: "torch.Tensor",
            non_visual: "torch.Tensor",
            actions: "torch.Tensor",
            lstm_state: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Used during PPO update.

            Returns (log_probs, values, entropy)
            """
            logits, value, _ = self.forward(vision, non_visual, lstm_state)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy   = dist.entropy()
            return log_probs, value, entropy


else:
    # Stub classes when torch is not installed (for tests / CI)
    class VisualEncoder:  # type: ignore[no-redef]
        out_dim = CNN_OUT_DIM

    class SensoryEncoder:  # type: ignore[no-redef]
        out_dim = 64

    class AttentionGate:  # type: ignore[no-redef]
        pass

    class TenantNetwork:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required. pip install torch")
