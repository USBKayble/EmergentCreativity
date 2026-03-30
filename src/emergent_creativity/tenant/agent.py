"""
agent.py
========
The Tenant – the RL agent that lives in the apartment.

The tenant is represented as a capsule rigid body in PyBullet.
On each step it:
  1. Reads sensory observations (via SensorySuite).
  2. Executes the action chosen by the neural network.
  3. Updates internal vitals (hunger, energy, bladder, happiness).
  4. Emits *events* that the reward system can listen to.

Events are plain strings stored in a list that is drained each step by
:class:`~emergent_creativity.rewards.ruleset.RewardEvaluator`.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..environment.objects import (
    ObjectRegistry,
    WorldObject,
)
from ..environment.physics_world import PhysicsWorld
from ..environment.senses import SensorySuite, SensoryObservation
from .actions import (
    Action,
    TURN_SPEED_DEG,
    REACH_DISTANCE,
    SLEEP_DISTANCE,
    TOILET_DISTANCE,
)

try:
    import pybullet as pb

    _PB = True
    LINK_FRAME = pb.LINK_FRAME if hasattr(pb, "LINK_FRAME") else 2
except ImportError:
    _PB = False
    pb = None  # type: ignore
    LINK_FRAME = 0

# ---------------------------------------------------------------------------
# Vitals container
# ---------------------------------------------------------------------------


@dataclass
class Vitals:
    """
    Scalar state variables tracking the tenant's biological needs.
    All values are in [0, 1] unless noted.
    """

    hunger: float = 0.0  # 0=full, 1=starving
    energy: float = 1.0  # 0=exhausted, 1=fully rested
    bladder: float = 0.0  # 0=empty, 1=urgent
    happiness: float = 0.5  # subjective mood

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.hunger, self.energy, self.bladder, self.happiness],
            dtype=np.float32,
        )

    def clamp(self) -> None:
        self.hunger = max(0.0, min(1.0, self.hunger))
        self.energy = max(0.0, min(1.0, self.energy))
        self.bladder = max(0.0, min(1.0, self.bladder))
        self.happiness = max(0.0, min(1.0, self.happiness))


# ---------------------------------------------------------------------------
# Tenant
# ---------------------------------------------------------------------------


class Tenant:
    """
    The RL agent that inhabits the apartment.

    Parameters
    ----------
    world      : PhysicsWorld
    registry   : ObjectRegistry
    sensors    : SensorySuite
    vitals_cfg : dict   (from rewards.yaml ``vitals`` section)
    spawn_pos  : (x, y, z)   initial position
    """

    CAPSULE_RADIUS = 0.25  # metres
    CAPSULE_HEIGHT = 1.2  # metres (cylinder part)
    AGENT_MASS = 70.0  # kg
    EYE_HEIGHT = 1.55  # metres above base

    def __init__(
        self,
        world: PhysicsWorld,
        registry: ObjectRegistry,
        sensors: SensorySuite,
        vitals_cfg: Optional[dict] = None,
        spawn_pos: Tuple[float, float, float] = (2.5, 2.5, 0.0),
    ) -> None:
        self._world = world
        self._registry = registry
        self._sensors = sensors
        self._cfg = vitals_cfg or {}

        self.vitals = Vitals()
        self.inventory: Optional[WorldObject] = None  # held item
        self._body_id = -1
        self._yaw = 0.0  # radians, current heading
        self._spawn_pos = spawn_pos

        # Event queue – drained by the reward evaluator each step
        self.events: List[str] = []

        # Step counters
        self.idle_steps = 0
        self.total_steps = 0
        self._is_sleeping = False

    # ------------------------------------------------------------------
    # Spawn / reset
    # ------------------------------------------------------------------

    def spawn(self) -> None:
        """Create the capsule body in the physics world."""
        pos = (
            self._spawn_pos[0],
            self._spawn_pos[1],
            self._spawn_pos[2] + self.CAPSULE_RADIUS + self.CAPSULE_HEIGHT / 2,
        )
        # Use a cylinder as the collision shape (capsule approximation)
        if not _PB:
            self._body_id = -1
            return
        col_id = pb.createCollisionShape(
            pb.GEOM_CAPSULE,
            radius=self.CAPSULE_RADIUS,
            height=self.CAPSULE_HEIGHT,
            physicsClientId=self._world.client,
        )
        vis_id = pb.createVisualShape(
            pb.GEOM_CAPSULE,
            radius=self.CAPSULE_RADIUS,
            length=self.CAPSULE_HEIGHT,
            rgbaColor=(0.4, 0.6, 0.9, 1.0),
            physicsClientId=self._world.client,
        )
        self._body_id = pb.createMultiBody(
            baseMass=self.AGENT_MASS,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            physicsClientId=self._world.client,
        )
        # Lock rotation to prevent the capsule from toppling over
        pb.changeDynamics(
            self._body_id,
            -1,
            angularDamping=1.0,
            linearDamping=0.05,
            physicsClientId=self._world.client,
        )
        self._sensors.set_agent_body(self._body_id)

    def reset(self) -> None:
        """Respawn agent and reset vitals."""
        if self._body_id >= 0 and _PB:
            try:
                pb.removeBody(self._body_id, physicsClientId=self._world.client)
            except Exception:
                pass
        self._body_id = -1
        self.vitals = Vitals()
        self.inventory = None
        self._yaw = 0.0
        self.events.clear()
        self.idle_steps = 0
        self.total_steps = 0
        self._is_sleeping = False
        self._sensors.reset()
        self.spawn()

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def yaw(self) -> float:
        return self._yaw

    def get_position(self) -> Tuple[float, float, float]:
        if self._body_id < 0 or not _PB:
            return self._spawn_pos
        pos, _ = self._world.get_position_orientation(self._body_id)
        # Return base position (subtract half height)
        return (pos[0], pos[1], pos[2] - self.CAPSULE_HEIGHT / 2 - self.CAPSULE_RADIUS)

    def get_eye_position(self) -> Tuple[float, float, float]:
        base = self.get_position()
        return (base[0], base[1], base[2] + self.EYE_HEIGHT)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> Tuple[SensoryObservation, np.ndarray]:
        """
        Returns
        -------
        sensory  : SensoryObservation  – raw five-sense data
        vitals   : np.ndarray shape (4,)  – hunger/energy/bladder/happiness
        """
        pos = self.get_position()
        sensory = self._sensors.observe(pos, self._yaw)
        return sensory, self.vitals.to_array()

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def step(self, action: int) -> None:
        """
        Execute one action and update internal state.
        Emits events into self.events.
        """
        if self._body_id < 0:
            return

        self.total_steps += 1
        self.events.clear()

        act = Action(action)

        if act == Action.IDLE:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        # --- Movement ---
        if act in (
            Action.MOVE_FORWARD,
            Action.MOVE_BACKWARD,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        ):
            self._move(act)
        elif act == Action.TURN_LEFT:
            self._yaw -= math.radians(TURN_SPEED_DEG)
            if self._body_id >= 0 and _PB:
                orn = pb.getQuaternionFromEuler([0, 0, self._yaw])
                pos, _ = pb.getBasePositionAndOrientation(
                    self._body_id, physicsClientId=self._world.client
                )
                pb.resetBasePositionAndOrientation(
                    self._body_id, pos, orn, physicsClientId=self._world.client
                )
        elif act == Action.TURN_RIGHT:
            self._yaw += math.radians(TURN_SPEED_DEG)
            if self._body_id >= 0 and _PB:
                orn = pb.getQuaternionFromEuler([0, 0, self._yaw])
                pos, _ = pb.getBasePositionAndOrientation(
                    self._body_id, physicsClientId=self._world.client
                )
                pb.resetBasePositionAndOrientation(
                    self._body_id, pos, orn, physicsClientId=self._world.client
                )

        # --- Object interaction ---
        elif act == Action.PICK_UP:
            self._pick_up()
        elif act == Action.PUT_DOWN:
            self._put_down()
        elif act == Action.INTERACT:
            self._interact()
        elif act == Action.EAT:
            self._eat()
        elif act == Action.SLEEP:
            self._sleep()
        elif act == Action.USE_BATHROOM:
            self._use_bathroom()

        # --- Vitals update ---
        self._update_vitals()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _move(self, act: Action) -> None:
        if self._body_id < 0 or not _PB:
            return
        if self._is_sleeping:
            self._is_sleeping = False

        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        speed = 2.0  # m/s

        if act == Action.MOVE_FORWARD:
            vx, vy = sin_y * speed, cos_y * speed
        elif act == Action.MOVE_BACKWARD:
            vx, vy = -sin_y * speed, -cos_y * speed
        elif act == Action.MOVE_LEFT:
            vx, vy = -cos_y * speed, sin_y * speed
        else:  # MOVE_RIGHT
            vx, vy = cos_y * speed, -sin_y * speed

        cur_pos = self.get_position()
        new_x = cur_pos[0] + vx * 0.016
        new_y = cur_pos[1] + vy * 0.016
        if new_x < 0.3 or new_x > 9.7 or new_y < 0.3 or new_y > 9.7:
            vx, vy = 0, 0

        lin_vel, ang_vel = pb.getBaseVelocity(
            self._body_id, physicsClientId=self._world.client
        )
        pb.resetBaseVelocity(
            self._body_id,
            linearVelocity=[vx, vy, lin_vel[2]],
            angularVelocity=[0, 0, 0],
            physicsClientId=self._world.client,
        )

    # ------------------------------------------------------------------
    # Object interactions
    # ------------------------------------------------------------------

    def _nearest_reachable(self, **kwargs) -> Optional[WorldObject]:
        pos = self.get_position()
        obj = self._registry.nearest(pos, **kwargs)
        if obj is None:
            return None
        dist = self._registry.distance(obj.body_id, pos)
        if dist <= REACH_DISTANCE:
            return obj
        return None

    def _pick_up(self) -> None:
        if self.inventory is not None:
            return  # already holding something
        obj = self._nearest_reachable(exclude_held=True)
        if obj and obj.can_pick_up and not obj.consumed:
            obj.held_by_agent = True
            obj.on_floor = False
            if obj.is_mess:
                self.events.append("picked_up_mess")
            self.inventory = obj

    def _put_down(self) -> None:
        if self.inventory is None:
            return
        obj = self.inventory
        # Find nearest surface or place on floor
        pos = self.get_position()
        surface = self._nearest_reachable(category=None, exclude_held=False)
        if surface and surface.is_surface and surface is not obj:
            # Place on top of surface
            surf_pos = self._registry.position_of(surface.body_id)
            if surf_pos:
                target_pos = (
                    surf_pos[0],
                    surf_pos[1],
                    surf_pos[2] + surface.half_extents[2] + obj.half_extents[2] + 0.02,
                )
                if obj.body_id >= 0 and _PB:
                    pb.resetBasePositionAndOrientation(
                        obj.body_id,
                        target_pos,
                        (0, 0, 0, 1),
                        physicsClientId=self._world.client,
                    )
                self.events.append("placed_item_on_surface")
                obj.on_floor = False
        else:
            # Drop on floor in front of agent
            cos_y = math.cos(self._yaw)
            sin_y = math.sin(self._yaw)
            drop_x = pos[0] + sin_y * 0.8
            drop_y = pos[1] + cos_y * 0.8
            if obj.body_id >= 0 and _PB:
                pb.resetBasePositionAndOrientation(
                    obj.body_id,
                    (drop_x, drop_y, obj.half_extents[2] + 0.02),
                    (0, 0, 0, 1),
                    physicsClientId=self._world.client,
                )
            obj.on_floor = True

        obj.held_by_agent = False
        self.inventory = None

    def _interact(self) -> None:
        obj = self._nearest_reachable(exclude_held=False)
        if obj is None or not obj.is_interactive:
            return
        obj.active = not obj.active
        if obj.name == "tv":
            if obj.active:
                self.events.append("watching_tv")
                self.vitals.happiness = min(1.0, self.vitals.happiness + 0.15)
        elif "book" in obj.name:
            if obj.active:
                self.events.append("reading_book")
                self.vitals.happiness = min(1.0, self.vitals.happiness + 0.1)
        elif "game" in obj.name:
            if obj.active:
                self.events.append("playing_game")
                self.vitals.happiness = min(1.0, self.vitals.happiness + 0.2)
        elif obj.name == "fridge":
            pass
        elif obj.name == "shower":
            if obj.active:
                self.events.append("showered")

    def _eat(self) -> None:
        if self.inventory is None:
            return
        obj = self.inventory
        if not obj.is_food or obj.consumed:
            return
        obj.consumed = True
        obj.held_by_agent = False
        self.inventory = None
        # Reduce hunger
        self.vitals.hunger = max(0.0, self.vitals.hunger - obj.nutrition)
        self.vitals.happiness = min(1.0, self.vitals.happiness + 0.05)
        # Trigger taste sense
        self._sensors.taste.activate(obj)
        self.events.append("ate_food")
        # Remove consumed food from physics world
        if obj.body_id >= 0 and _PB:
            try:
                self._world.remove_body(obj.body_id)
                self._registry.unregister(obj.body_id)
            except Exception:
                pass

    def _sleep(self) -> None:
        pos = self.get_position()
        # Find nearest bed or sofa
        for obj in self._registry.objects_within_radius(pos, SLEEP_DISTANCE):
            if obj.name in ("bed", "sofa") and obj.is_interactive:
                self._is_sleeping = True
                self.events.append("sleeping")
                return

    def _use_bathroom(self) -> None:
        pos = self.get_position()
        for obj in self._registry.objects_within_radius(pos, TOILET_DISTANCE):
            if obj.name == "toilet":
                self.vitals.bladder = max(0.0, self.vitals.bladder - 0.9)
                self.events.append("used_bathroom")
                return
        # Not near toilet - penalty for failed attempt
        self.events.append("failed_bathroom_attempt")

    # ------------------------------------------------------------------
    # Vitals update
    # ------------------------------------------------------------------

    def _update_vitals(self) -> None:
        cfg = self._cfg
        # Hunger increases every step
        self.vitals.hunger += cfg.get("hunger_rate", 0.0002)
        # Energy drains unless sleeping
        if self._is_sleeping:
            self.vitals.energy += cfg.get("energy_regen", 0.0005)
            # happiness improves slightly while sleeping
            self.vitals.happiness += 0.0001
        else:
            self.vitals.energy -= cfg.get("energy_drain", 0.0001)
        # Bladder fills
        self.vitals.bladder += cfg.get("bladder_rate", 0.00015)
        # Happiness decays slowly
        self.vitals.happiness -= cfg.get("happiness_decay", 0.00005)
        self.vitals.clamp()

    # ------------------------------------------------------------------
    # Convenience properties for reward evaluator
    # ------------------------------------------------------------------

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    @property
    def is_watching_tv(self) -> bool:
        for obj in self._registry.all():
            if obj.name == "tv" and obj.active:
                dist = self._registry.distance(obj.body_id, self.get_position())
                if dist < 4.0:
                    return True
        return False

    @property
    def is_reading(self) -> bool:
        return (
            self.inventory is not None
            and "book" in self.inventory.name
            and self.inventory.active
        )

    @property
    def is_playing_game(self) -> bool:
        return (
            self.inventory is not None
            and "game" in self.inventory.name
            and self.inventory.active
        )
