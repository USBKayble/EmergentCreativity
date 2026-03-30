"""
sim_env.py
==========
Gymnasium-compatible environment wrapping the full apartment simulation.

This is the primary interface for RL training.  The observation space
contains all five senses plus vitals; the action space is discrete
(see :mod:`emergent_creativity.tenant.actions`).

Observation space
-----------------
::

    {
        "vision"    : Box(0, 1, shape=(H, W, 3),           dtype=float32)
        "hearing"   : Box(0, 1, shape=(HEARING_DIM,),       dtype=float32)
        "touch"     : Box(0, 1, shape=(TOUCH_DIM,),         dtype=float32)
        "smell"     : Box(0, 1, shape=(SMELL_DIM,),         dtype=float32)
        "taste"     : Box(0, 1, shape=(TASTE_DIM,),         dtype=float32)
        "vitals"    : Box(0, 1, shape=(4,),                 dtype=float32)
    }

Action space
------------
::

    Discrete(13)   # see emergent_creativity.tenant.actions.Action

Example
-------
::

    env = TenantEnv(gui=False)
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces

        _GYM = True
    except ImportError:
        _GYM = False

from .environment.physics_world import PhysicsWorld, CameraSpec, PhysicsConfig
from .environment.objects import ObjectRegistry
from .environment.apartment import Apartment
from .environment.senses import (
    SensorySuite,
    VISION_H,
    VISION_W,
    VISION_C,
    HEARING_DIM,
    TOUCH_DIM,
    SMELL_DIM,
    TASTE_DIM,
)
from .tenant.agent import Tenant
from .tenant.actions import N_ACTIONS
from .rewards.ruleset import RewardEvaluator

DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "rewards.yaml"


def _require_gym() -> None:
    if not _GYM:
        raise ImportError("gymnasium (or gym) is required. pip install gymnasium")


class TenantEnv:
    """
    The main simulation environment.

    Parameters
    ----------
    gui         : bool   – open PyBullet GUI window
    config_path : str    – path to rewards.yaml
    seed        : int    – random seed for reproducibility
    render_mode : str    – "human" or "rgb_array" (for Gymnasium compatibility)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        gui: bool = False,
        config_path: Optional[str] = None,
        seed: int = 42,
        render_mode: str = "rgb_array",
    ) -> None:
        _require_gym()

        self._gui = gui
        self._seed = seed
        self.render_mode = render_mode

        # Load reward config
        cfg_path = str(config_path or DEFAULT_CONFIG)
        self._evaluator = RewardEvaluator.from_yaml(cfg_path)
        obs_cfg = {}
        # Read observation settings from YAML
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            obs_cfg = raw.get("observation", {})
        except Exception:
            pass

        w = obs_cfg.get("vision_width", VISION_W)
        h = obs_cfg.get("vision_height", VISION_H)

        # Gymnasium spaces
        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(
                    0.0, 1.0, shape=(h, w, VISION_C), dtype=np.float32
                ),
                "hearing": spaces.Box(0.0, 1.0, shape=(HEARING_DIM,), dtype=np.float32),
                "touch": spaces.Box(0.0, 1.0, shape=(TOUCH_DIM,), dtype=np.float32),
                "smell": spaces.Box(0.0, 1.0, shape=(SMELL_DIM,), dtype=np.float32),
                "taste": spaces.Box(0.0, 1.0, shape=(TASTE_DIM,), dtype=np.float32),
                "vitals": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal components (created on reset)
        self._world: Optional[PhysicsWorld] = None
        self._registry: Optional[ObjectRegistry] = None
        self._apartment: Optional[Apartment] = None
        self._sensors: Optional[SensorySuite] = None
        self._tenant: Optional[Tenant] = None

        self._initialised = False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """Reset the environment and return the first observation."""
        _seed = seed if seed is not None else self._seed

        if not self._initialised:
            self._build(_seed)

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """Apply *action*, advance simulation by one step."""
        assert self._initialised, "Call reset() first."

        # Execute action
        self._tenant.step(action)

        # Advance physics
        self._world.step()

        # Sync object positions
        self._apartment.sync_registry()

        # Compute reward
        reward, reward_info = self._evaluator.evaluate(self._tenant, self._registry)

        # Check termination
        terminated = self._evaluator.is_terminal(self._tenant)
        truncated = False

        obs = self._get_obs()
        info = {
            "reward_breakdown": reward_info,
            "vitals": self._tenant.vitals.to_array().tolist(),
            "events": list(self._tenant.events),
            "mess_count": self._registry.mess_count(),
            "step": self._tenant.total_steps,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current view from the tenant's perspective."""
        if self._tenant is None:
            return None
        pos = self._tenant.get_position()
        yaw = self._tenant.yaw
        return self._sensors.vision.observe(pos, yaw)

    def close(self) -> None:
        """Shut down the physics engine."""
        if self._world is not None:
            self._world.stop()
            self._world = None
        self._initialised = False

    @property
    def physics(self) -> Optional[PhysicsWorld]:
        return self._world

    # ------------------------------------------------------------------
    # Internal build helpers
    # ------------------------------------------------------------------

    def _build(self, seed: int) -> None:
        """First-time initialisation."""
        import yaml
        from pathlib import Path as _P

        cfg_path = str(DEFAULT_CONFIG)
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            obs_cfg = raw.get("observation", {})
            vitals_cfg = raw.get("vitals", {})
        except Exception:
            obs_cfg = {}
            vitals_cfg = {}

        # Physics world
        cam = CameraSpec(
            width=obs_cfg.get("vision_width", VISION_W),
            height=obs_cfg.get("vision_height", VISION_H),
            fov=obs_cfg.get("vision_fov", 90.0),
        )
        self._world = PhysicsWorld(gui=self._gui, camera=cam)
        self._world.start()

        self._registry = ObjectRegistry()
        self._apartment = Apartment(self._world, self._registry, seed=seed)
        self._apartment.build()

        self._sensors = SensorySuite(
            self._world, self._registry, agent_body_id=-1, cfg=obs_cfg
        )

        self._tenant = Tenant(
            self._world,
            self._registry,
            self._sensors,
            vitals_cfg=vitals_cfg,
        )
        self._tenant.spawn()

        # Run a few steps to settle the physics
        for _ in range(10):
            self._world.step()
        self._apartment.sync_registry()

        self._initialised = True

    def _rebuild(self, seed: int) -> None:
        """Reset for a new episode (reuse physics world, respawn items)."""
        self._apartment.reset_items()
        self._tenant.reset()
        for _ in range(10):
            self._world.step()
        self._apartment.sync_registry()

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        sensory, vitals = self._tenant.observe()
        return {
            "vision": sensory.vision,
            "hearing": sensory.hearing,
            "touch": sensory.touch,
            "smell": sensory.smell,
            "taste": sensory.taste,
            "vitals": vitals,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tenant(self) -> Optional[Tenant]:
        return self._tenant

    @property
    def registry(self) -> Optional[ObjectRegistry]:
        return self._registry

    @property
    def apartment(self) -> Optional[Apartment]:
        return self._apartment

    @property
    def evaluator(self) -> RewardEvaluator:
        return self._evaluator
