"""
viewer.py
=========
Real-time interactive viewer for the EmergentCreativity simulation.

Features
--------
* Displays the tenant's first-person camera view (left panel).
* Overlays live vitals (hunger, energy, bladder, happiness) as coloured bars.
* Shows last action label, current reward, total reward, and episode step.
* Right panel: top-down minimap of the apartment (using 2-D overhead render).
* Reward history graph (rolling 200-step window).
* **User controls** (keyboard):
    - SPACE      → pause/resume the simulation
    - R          → reset the episode
    - ↑ ↓ ← →   → manually override action (move forward/back/left/right)
    - A / D      → turn left / right
    - E          → interact
    - F          → pick up
    - G          → put down
    - T          → eat
    - S          → sleep
    - B          → use bathroom
    - I          → toggle agent control (manual vs. NN)
    - Q / ESC    → quit

The viewer runs the simulation in its own loop at up to *target_fps* Hz.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import pygame
    _PG = True
except ImportError:
    _PG = False

from ..sim_env import TenantEnv
from ..tenant.actions import Action, ACTION_LABELS, N_ACTIONS

# Window layout
WIN_W  = 1280
WIN_H  = 720
VIEW_W = 640   # first-person view width
VIEW_H = 480   # first-person view height
MAP_W  = 300
MAP_H  = 300
PANEL_X = 660  # right panel x offset

# Colour palette
BG_COLOR       = (20,  20,  30)
TEXT_COLOR     = (220, 220, 220)
ACCENT_COLOR   = (80,  160, 255)
WARN_COLOR     = (255, 180,  60)
DANGER_COLOR   = (220,  60,  60)
GOOD_COLOR     = (80,  200, 120)
GRAPH_COLOR    = (100, 160, 255)
GRID_COLOR     = (50,  50,  70)
OVERLAY_BG     = (0,   0,   0, 160)

VITAL_COLORS = {
    "hunger":    DANGER_COLOR,
    "energy":    GOOD_COLOR,
    "bladder":   WARN_COLOR,
    "happiness": ACCENT_COLOR,
}

MANUAL_ACTION_MAP = {
    pygame.K_UP:    Action.MOVE_FORWARD   if _PG else 1,
    pygame.K_DOWN:  Action.MOVE_BACKWARD  if _PG else 2,
    pygame.K_LEFT:  Action.MOVE_LEFT      if _PG else 3,
    pygame.K_RIGHT: Action.MOVE_RIGHT     if _PG else 4,
    pygame.K_a:     Action.TURN_LEFT      if _PG else 5,
    pygame.K_d:     Action.TURN_RIGHT     if _PG else 6,
    pygame.K_f:     Action.PICK_UP        if _PG else 7,
    pygame.K_g:     Action.PUT_DOWN       if _PG else 8,
    pygame.K_e:     Action.INTERACT       if _PG else 9,
    pygame.K_t:     Action.EAT            if _PG else 10,
    pygame.K_s:     Action.SLEEP          if _PG else 11,
    pygame.K_b:     Action.USE_BATHROOM   if _PG else 12,
} if _PG else {}


def _require_pygame() -> None:
    if not _PG:
        raise ImportError("pygame is required. pip install pygame")


class SimViewer:
    """
    Interactive viewer for the EmergentCreativity apartment simulation.

    Parameters
    ----------
    env        : TenantEnv  – already-initialised (or will be reset here)
    nn_agent   : optional callable (obs → action)  – if provided, used in NN mode
    target_fps : int
    """

    def __init__(
        self,
        env: TenantEnv,
        nn_agent=None,
        target_fps: int = 30,
    ) -> None:
        _require_pygame()
        self.env        = env
        self.nn_agent   = nn_agent
        self.target_fps = target_fps

        self._paused        = False
        self._manual_mode   = nn_agent is None
        self._manual_action = Action.IDLE
        self._running       = True

        self._total_reward   = 0.0
        self._step_reward    = 0.0
        self._episode_step   = 0
        self._episode_count  = 0
        self._reward_history: deque = deque(maxlen=200)
        self._action_label   = ACTION_LABELS[Action.IDLE]
        self._last_info: dict = {}

        pygame.init()
        self._screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("EmergentCreativity – Apartment Simulation")
        self._clock = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("monospace", 18)
        self._font_sm = pygame.font.SysFont("monospace", 14)
        self._font_xs = pygame.font.SysFont("monospace", 12)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the viewer main loop (blocks until quit)."""
        obs, _ = self.env.reset()
        lstm_state = None

        while self._running:
            # --- Event handling ---
            self._manual_action = Action.IDLE
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)

            if self._paused:
                self._render(obs)
                self._clock.tick(10)
                continue

            # --- Determine action ---
            if self._manual_mode:
                # Check held keys for smooth movement
                keys = pygame.key.get_pressed()
                action = Action.IDLE
                for k, a in MANUAL_ACTION_MAP.items():
                    if keys[k]:
                        action = a
                        break
                if self._manual_action != Action.IDLE:
                    action = self._manual_action
            else:
                # NN agent chooses action
                action = self._nn_act(obs, lstm_state)

            self._action_label = ACTION_LABELS.get(int(action), "?")

            # --- Step environment ---
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            self._step_reward    = reward
            self._total_reward  += reward
            self._episode_step  += 1
            self._last_info      = info
            self._reward_history.append(reward)

            if terminated or truncated:
                self._episode_count += 1
                self._total_reward   = 0.0
                self._episode_step   = 0
                obs, _ = self.env.reset()
                lstm_state = None

            # --- Render ---
            self._render(obs)
            self._clock.tick(self.target_fps)

        pygame.quit()

    def _nn_act(self, obs, lstm_state):
        """Call the NN agent callable (if provided)."""
        if self.nn_agent is None:
            return Action.IDLE
        try:
            result = self.nn_agent(obs, lstm_state)
            if isinstance(result, tuple):
                return result[0]
            return result
        except Exception:
            return Action.IDLE

    def _handle_keydown(self, key: int) -> None:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self._running = False
        elif key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key == pygame.K_r:
            self.env.reset()
            self._total_reward  = 0.0
            self._episode_step  = 0
        elif key == pygame.K_i:
            self._manual_mode = not self._manual_mode
            mode = "Manual" if self._manual_mode else "NN"
            print(f"[Viewer] Switched to {mode} control")
        else:
            for k, a in MANUAL_ACTION_MAP.items():
                if key == k:
                    self._manual_action = a
                    break

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, obs: dict) -> None:
        self._screen.fill(BG_COLOR)
        self._draw_first_person_view(obs)
        self._draw_minimap()
        self._draw_vitals(obs)
        self._draw_hud()
        self._draw_reward_graph()
        pygame.display.flip()

    def _draw_first_person_view(self, obs: dict) -> None:
        vision = obs.get("vision")
        if vision is None:
            return
        # vision is (H, W, 3) float32 [0,1]
        rgb_u8 = (vision * 255).astype(np.uint8)
        # pygame wants (W, H) array — transpose
        surf = pygame.surfarray.make_surface(rgb_u8.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (VIEW_W, VIEW_H))
        self._screen.blit(surf, (10, 10))

        # Label
        lbl = self._font_sm.render("First-Person View", True, TEXT_COLOR)
        self._screen.blit(lbl, (10, VIEW_H + 16))

    def _draw_minimap(self) -> None:
        """Render a simple top-down minimap of the apartment."""
        apt = self.env.apartment
        if apt is None:
            return

        map_surf = pygame.Surface((MAP_W, MAP_H))
        map_surf.fill((30, 40, 30))

        # Apartment is 10m × 10m; scale to MAP_W × MAP_H
        scale_x = MAP_W / 10.0
        scale_y = MAP_H / 10.0

        def world_to_map(wx, wy):
            return int(wx * scale_x), int(wy * scale_y)

        # Draw grid lines for rooms
        for gx in [5.0]:
            sx, _ = world_to_map(gx, 0)
            pygame.draw.line(map_surf, GRID_COLOR, (sx, 0), (sx, MAP_H), 2)
        for gy in [5.0]:
            _, sy = world_to_map(0, gy)
            pygame.draw.line(map_surf, GRID_COLOR, (0, sy), (MAP_W, sy), 2)

        # Draw objects
        registry = self.env.registry
        if registry:
            for obj in registry.all():
                pos = registry.position_of(obj.body_id)
                if pos is None:
                    continue
                mx, my = world_to_map(pos[0], pos[1])
                if obj.category.name == "FURNITURE":
                    color = (120, 100, 80)
                elif obj.is_food:
                    color = (100, 200, 100)
                elif obj.is_mess:
                    color = (200, 150, 50)
                else:
                    color = (180, 180, 220)
                r = max(2, int(min(obj.half_extents[:2]) * scale_x))
                pygame.draw.circle(map_surf, color, (mx, my), r)

        # Draw tenant
        tenant = self.env.tenant
        if tenant:
            tpos = tenant.get_position()
            tx, ty = world_to_map(tpos[0], tpos[1])
            pygame.draw.circle(map_surf, (80, 160, 255), (tx, ty), 6)
            # Draw heading arrow
            yaw = tenant.yaw
            arrow_len = 12
            ex = int(tx + math.sin(yaw) * arrow_len)
            ey = int(ty + math.cos(yaw) * arrow_len)
            pygame.draw.line(map_surf, (255, 255, 100), (tx, ty), (ex, ey), 2)

        # Border
        pygame.draw.rect(map_surf, ACCENT_COLOR, (0, 0, MAP_W, MAP_H), 2)

        self._screen.blit(map_surf, (PANEL_X, 10))
        lbl = self._font_sm.render("Minimap (top-down)", True, TEXT_COLOR)
        self._screen.blit(lbl, (PANEL_X, MAP_H + 16))

    def _draw_vitals(self, obs: dict) -> None:
        """Draw vital bars below the minimap."""
        vitals_arr = obs.get("vitals", np.zeros(4, dtype=np.float32))
        labels = ["Hunger", "Energy", "Bladder", "Happiness"]
        colors  = [DANGER_COLOR, GOOD_COLOR, WARN_COLOR, ACCENT_COLOR]

        bar_x = PANEL_X
        bar_y = MAP_H + 40
        bar_w = MAP_W
        bar_h = 18
        gap   = 28

        for i, (label, color, val) in enumerate(zip(labels, colors, vitals_arr)):
            y = bar_y + i * gap
            # Background
            pygame.draw.rect(self._screen, (40, 40, 60), (bar_x, y, bar_w, bar_h), border_radius=4)
            # Fill – hunger & bladder: high is bad; energy & happiness: high is good
            if label in ("Hunger", "Bladder"):
                fill_frac = float(val)
                fill_color = _lerp_color((80, 180, 80), color, fill_frac)
            else:
                fill_frac = float(val)
                fill_color = _lerp_color(color, (180, 80, 80), 1.0 - fill_frac)
            fill_w = max(0, int(bar_w * fill_frac))
            if fill_w > 0:
                pygame.draw.rect(
                    self._screen, fill_color,
                    (bar_x, y, fill_w, bar_h), border_radius=4
                )
            # Label + value
            text = self._font_xs.render(f"{label}: {val:.2f}", True, TEXT_COLOR)
            self._screen.blit(text, (bar_x + 4, y + 2))

    def _draw_hud(self) -> None:
        """Draw HUD text on the right side."""
        hud_x = PANEL_X
        hud_y = MAP_H + 160
        line_h = 22

        mode_str = "MANUAL" if self._manual_mode else "NN AGENT"
        paused_str = " [PAUSED]" if self._paused else ""

        lines = [
            f"Mode: {mode_str}{paused_str}",
            f"Episode: {self._episode_count}",
            f"Step:    {self._episode_step}",
            f"Action:  {self._action_label}",
            f"Reward:  {self._step_reward:+.3f}",
            f"Total:   {self._total_reward:+.2f}",
            f"Mess:    {self._last_info.get('mess_count', '?')}",
        ]

        for i, line in enumerate(lines):
            surf = self._font_sm.render(line, True, TEXT_COLOR)
            self._screen.blit(surf, (hud_x, hud_y + i * line_h))

        # Controls hint
        hint_y = WIN_H - 110
        hints = [
            "SPACE=Pause  R=Reset  I=Toggle NN/Manual  Q=Quit",
            "↑↓←→=Move  A/D=Turn  E=Interact  F=Pick  G=Drop",
            "T=Eat  S=Sleep  B=Bathroom",
        ]
        for i, h in enumerate(hints):
            s = self._font_xs.render(h, True, (150, 150, 180))
            self._screen.blit(s, (10, hint_y + i * 18))

    def _draw_reward_graph(self) -> None:
        """Small rolling reward graph at the bottom-left."""
        if len(self._reward_history) < 2:
            return

        graph_x, graph_y = 10, VIEW_H + 40
        graph_w, graph_h = VIEW_W, 120
        pygame.draw.rect(self._screen, (25, 25, 40), (graph_x, graph_y, graph_w, graph_h))
        pygame.draw.rect(self._screen, ACCENT_COLOR,  (graph_x, graph_y, graph_w, graph_h), 1)

        data = list(self._reward_history)
        max_v = max(abs(v) for v in data) + 1e-8
        zero_y = graph_y + graph_h // 2

        pygame.draw.line(
            self._screen, GRID_COLOR,
            (graph_x, zero_y), (graph_x + graph_w, zero_y), 1
        )

        pts = []
        for i, v in enumerate(data):
            px = graph_x + int(i / (len(data) - 1) * (graph_w - 2)) + 1
            py = zero_y - int(v / max_v * (graph_h // 2 - 4))
            py = max(graph_y + 2, min(graph_y + graph_h - 2, py))
            pts.append((px, py))

        if len(pts) >= 2:
            pygame.draw.lines(self._screen, GRAPH_COLOR, False, pts, 2)

        lbl = self._font_xs.render("Step Reward History", True, TEXT_COLOR)
        self._screen.blit(lbl, (graph_x + 4, graph_y + 4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp_color(c1, c2, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
