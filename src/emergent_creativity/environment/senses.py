"""
senses.py
=========
Implements all five human senses for the tenant agent.

Each sense produces a numerical observation that becomes part of the NN
input vector.  The senses are designed to be computationally cheap so the
simulation can run in real time even when rendering at ~60 Hz.

Sense breakdown
---------------
1. **Vision**   – rendered camera image (H×W×3 uint8 → float32 [0,1])
2. **Hearing**  – per-object sound level weighted by distance (float32 vector)
3. **Touch**    – contact-force magnitudes at the agent's collision shape
4. **Smell**    – nearby smell intensities per SmellType (float32 vector)
5. **Taste**    – active only when eating; returns taste vector (float32)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .objects import ObjectRegistry, SmellType, TasteType, WorldObject
from .physics_world import PhysicsWorld


# ---------------------------------------------------------------------------
# Observation shapes (used to define Gymnasium spaces)
# ---------------------------------------------------------------------------

VISION_H = 84
VISION_W = 84
VISION_C = 3                          # RGB channels
HEARING_DIM   = 8                     # one slot per "sound source category"
TOUCH_DIM     = 4                     # [front, back, left, right] force
SMELL_DIM     = len(SmellType)        # one slot per smell type
TASTE_DIM     = len(TasteType)        # one slot per taste type

TOTAL_SENSORY_DIM = (
    HEARING_DIM + TOUCH_DIM + SMELL_DIM + TASTE_DIM
)  # non-visual flattened length


@dataclass
class SensoryObservation:
    """
    Container for a single-timestep sensory reading.

    vision   : np.ndarray shape (H, W, 3) dtype float32, values in [0, 1]
    hearing  : np.ndarray shape (HEARING_DIM,) – normalised sound levels
    touch    : np.ndarray shape (TOUCH_DIM,)   – normalised contact forces
    smell    : np.ndarray shape (SMELL_DIM,)   – normalised smell intensities
    taste    : np.ndarray shape (TASTE_DIM,)   – normalised taste intensities
    """
    vision:  np.ndarray = field(
        default_factory=lambda: np.zeros((VISION_H, VISION_W, VISION_C), dtype=np.float32)
    )
    hearing: np.ndarray = field(
        default_factory=lambda: np.zeros(HEARING_DIM, dtype=np.float32)
    )
    touch:   np.ndarray = field(
        default_factory=lambda: np.zeros(TOUCH_DIM, dtype=np.float32)
    )
    smell:   np.ndarray = field(
        default_factory=lambda: np.zeros(SMELL_DIM, dtype=np.float32)
    )
    taste:   np.ndarray = field(
        default_factory=lambda: np.zeros(TASTE_DIM, dtype=np.float32)
    )

    def to_flat_non_visual(self) -> np.ndarray:
        """Concatenate all non-visual senses into a single vector."""
        return np.concatenate([self.hearing, self.touch, self.smell, self.taste])


# ---------------------------------------------------------------------------
# Individual sense implementations
# ---------------------------------------------------------------------------

class VisionSense:
    """
    Renders the scene from the agent's eye position.

    The eye is placed at the agent's head height (agent_z + eye_offset),
    looking in the direction the agent faces (yaw angle).
    """

    EYE_HEIGHT_OFFSET = 1.55  # metres above agent base (eye level for ~1.75m person)

    def __init__(
        self,
        world: PhysicsWorld,
        width: int = VISION_W,
        height: int = VISION_H,
        fov: float = 90.0,
    ) -> None:
        self._world = world
        self.width = width
        self.height = height
        self.fov = fov

    def observe(
        self,
        agent_pos: Tuple[float, float, float],
        agent_yaw: float,
    ) -> np.ndarray:
        """
        Return (H, W, 3) float32 image in [0, 1].

        Parameters
        ----------
        agent_pos : (x, y, z) base position of the agent.
        agent_yaw : heading in radians (0 = +Y axis).
        """
        ex = agent_pos[0]
        ey = agent_pos[1]
        ez = agent_pos[2] + self.EYE_HEIGHT_OFFSET

        # Look-at target 1 m ahead
        tx = ex + math.sin(agent_yaw)
        ty = ey + math.cos(agent_yaw)
        tz = ez

        rgb = self._world.render_camera(
            eye=(ex, ey, ez),
            target=(tx, ty, tz),
            up=(0.0, 0.0, 1.0),
            width=self.width,
            height=self.height,
            fov=self.fov,
        )
        return rgb.astype(np.float32) / 255.0


class HearingSense:
    """
    Simulates hearing by computing how loudly each sound-producing object
    is perceived at the agent's position (inverse-square attenuation).

    Output is a vector of length HEARING_DIM representing different audio
    "channels" (TV, appliances, water, voice, nature, alarm, footstep, ambient).
    """

    CHANNELS = [
        "tv_audio",
        "appliance_hum",
        "running_water",
        "voice",
        "ambient",
        "alarm",
        "footstep",
        "other",
    ]
    MAX_HEARING_DISTANCE = 15.0  # metres

    def __init__(self, registry: ObjectRegistry) -> None:
        self._registry = registry

    def observe(
        self,
        agent_pos: Tuple[float, float, float],
    ) -> np.ndarray:
        levels = np.zeros(HEARING_DIM, dtype=np.float32)
        for obj in self._registry.all():
            if obj.sensory.sound_level <= 0:
                continue
            dist = self._registry.distance(obj.body_id, agent_pos)
            if dist > self.MAX_HEARING_DISTANCE:
                continue
            # Inverse-square attenuation, clamped to 1 m minimum
            attenuation = 1.0 / max(1.0, dist ** 2)
            perceived = obj.sensory.sound_level * attenuation / 100.0  # 0-1 scale
            label = obj.sensory.sound_label
            if label in self.CHANNELS:
                idx = self.CHANNELS.index(label)
            else:
                idx = self.CHANNELS.index("other")
            levels[idx] = min(1.0, levels[idx] + perceived)
        return levels


class TouchSense:
    """
    Reports contact forces on four "sides" of the agent body.

    Uses PyBullet contact-point queries between the agent body and any
    other object.  Forces are normalised to [0, 1].
    """

    MAX_FORCE = 500.0  # Newtons

    def __init__(self, world: PhysicsWorld, agent_body_id: int) -> None:
        self._world = world
        self._agent_id = agent_body_id

    def set_agent_body(self, body_id: int) -> None:
        self._agent_id = body_id

    def observe(
        self,
        agent_pos: Tuple[float, float, float],
        agent_yaw: float,
    ) -> np.ndarray:
        """Return (4,) force vector [front, back, left, right], each [0,1]."""
        forces = np.zeros(TOUCH_DIM, dtype=np.float32)
        if self._agent_id < 0:
            return forces

        contacts = self._world.get_contact_points(self._agent_id)
        if not contacts:
            return forces

        for contact in contacts:
            # contact[7] = contact normal on object B in world space
            # contact[9] = normal force magnitude
            try:
                normal = contact[7]    # (nx, ny, nz)
                force_mag = contact[9]  # float
            except (IndexError, TypeError):
                continue

            if force_mag <= 0:
                continue

            # Project normal to agent's local frame
            cos_yaw = math.cos(agent_yaw)
            sin_yaw = math.sin(agent_yaw)
            local_x = normal[0] * cos_yaw - normal[1] * sin_yaw
            local_y = normal[0] * sin_yaw + normal[1] * cos_yaw

            norm_force = min(1.0, force_mag / self.MAX_FORCE)

            if local_y > 0.5:       # front
                forces[0] = min(1.0, forces[0] + norm_force)
            elif local_y < -0.5:    # back
                forces[1] = min(1.0, forces[1] + norm_force)
            elif local_x < -0.5:    # left
                forces[2] = min(1.0, forces[2] + norm_force)
            else:                   # right
                forces[3] = min(1.0, forces[3] + norm_force)

        return forces


class SmellSense:
    """
    Accumulates smell intensities per SmellType within the agent's smell
    radius.  Uses a simple linear decay with distance.
    """

    def __init__(
        self,
        registry: ObjectRegistry,
        radius: float = 3.0,
    ) -> None:
        self._registry = registry
        self.radius = radius

    def observe(
        self,
        agent_pos: Tuple[float, float, float],
    ) -> np.ndarray:
        levels = np.zeros(SMELL_DIM, dtype=np.float32)
        smell_types = list(SmellType)
        nearby = self._registry.objects_within_radius(agent_pos, self.radius)
        for obj in nearby:
            if obj.sensory.smell_intensity <= 0:
                continue
            dist = self._registry.distance(obj.body_id, agent_pos)
            decay = max(0.0, 1.0 - dist / self.radius)
            perceived = obj.sensory.smell_intensity * decay
            idx = smell_types.index(obj.sensory.smell_type)
            levels[idx] = min(1.0, levels[idx] + perceived)
        return levels


class TasteSense:
    """
    Produces a taste signal only when the agent is actively eating
    (i.e. holding a food item and the EAT action is triggered).

    Call :meth:`activate` when eating begins; the signal fades over
    ``duration`` steps.
    """

    FADE_STEPS = 20

    def __init__(self) -> None:
        self._active_tastes: np.ndarray = np.zeros(TASTE_DIM, dtype=np.float32)
        self._remaining_steps: int = 0

    def activate(self, obj: WorldObject) -> None:
        """Trigger taste sensation from eating *obj*."""
        taste_types = list(TasteType)
        vec = np.zeros(TASTE_DIM, dtype=np.float32)
        if obj.sensory.taste_intensity > 0:
            idx = taste_types.index(obj.sensory.taste_type)
            vec[idx] = obj.sensory.taste_intensity
        self._active_tastes = vec
        self._remaining_steps = self.FADE_STEPS

    def observe(self) -> np.ndarray:
        """Return current taste vector; fades over time."""
        if self._remaining_steps <= 0:
            return np.zeros(TASTE_DIM, dtype=np.float32)
        decay = self._remaining_steps / self.FADE_STEPS
        self._remaining_steps -= 1
        return (self._active_tastes * decay).astype(np.float32)

    def reset(self) -> None:
        self._active_tastes[:] = 0.0
        self._remaining_steps = 0


# ---------------------------------------------------------------------------
# Composite sensor – aggregates all five senses
# ---------------------------------------------------------------------------

class SensorySuite:
    """
    Bundles all five senses into a single interface.

    Parameters
    ----------
    world      : PhysicsWorld
    registry   : ObjectRegistry
    agent_body_id : int   (PyBullet body id of the tenant capsule)
    cfg        : dict     (from rewards.yaml observation section)
    """

    def __init__(
        self,
        world: PhysicsWorld,
        registry: ObjectRegistry,
        agent_body_id: int = -1,
        cfg: Optional[dict] = None,
    ) -> None:
        cfg = cfg or {}
        w = cfg.get("vision_width", VISION_W)
        h = cfg.get("vision_height", VISION_H)
        fov = cfg.get("vision_fov", 90.0)
        smell_r = cfg.get("smell_radius", 3.0)

        self.vision  = VisionSense(world, width=w, height=h, fov=fov)
        self.hearing = HearingSense(registry)
        self.touch   = TouchSense(world, agent_body_id)
        self.smell   = SmellSense(registry, radius=smell_r)
        self.taste   = TasteSense()

    def set_agent_body(self, body_id: int) -> None:
        self.touch.set_agent_body(body_id)

    def observe(
        self,
        agent_pos: Tuple[float, float, float],
        agent_yaw: float,
    ) -> SensoryObservation:
        """Gather all senses and return a :class:`SensoryObservation`."""
        return SensoryObservation(
            vision  = self.vision.observe(agent_pos, agent_yaw),
            hearing = self.hearing.observe(agent_pos),
            touch   = self.touch.observe(agent_pos, agent_yaw),
            smell   = self.smell.observe(agent_pos),
            taste   = self.taste.observe(),
        )

    def reset(self) -> None:
        self.taste.reset()
