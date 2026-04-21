"""
physics_world.py
================
Thin wrapper around the PyBullet physics engine.

Responsibilities
----------------
* Start / stop the bullet simulation in either GUI or DIRECT (headless) mode.
* Expose helpers for:
  - loading URDF / primitive collision shapes,
  - stepping the simulation,
  - querying contact points,
  - rendering camera images that the tenant's visual sense can consume.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pybullet as pb
    import pybullet_data
    _PYBULLET_AVAILABLE = True
except ImportError:
    pb = None  # type: ignore[assignment]
    pybullet_data = None  # type: ignore[assignment]
    _PYBULLET_AVAILABLE = False


@dataclass
class CameraSpec:
    """Parameters for a rendered camera view."""
    width: int = 84
    height: int = 84
    fov: float = 90.0          # vertical field-of-view in degrees
    near: float = 0.05
    far: float = 50.0


@dataclass
class PhysicsConfig:
    """Tunable physics-world parameters."""
    gravity: float = -9.81
    time_step: float = 1.0 / 60.0
    solver_iterations: int = 10
    use_real_time: bool = False


class PhysicsWorld:
    """
    Manages a single PyBullet physics instance.

    Parameters
    ----------
    gui : bool
        If *True* open the PyBullet GUI window; otherwise run headless.
    config : PhysicsConfig
        Simulation parameters.
    camera : CameraSpec
        Default camera parameters used for vision rendering.
    """

    def __init__(
        self,
        gui: bool = False,
        config: Optional[PhysicsConfig] = None,
        camera: Optional[CameraSpec] = None,
    ) -> None:
        if not _PYBULLET_AVAILABLE:
            raise ImportError(
                "pybullet is required for PhysicsWorld. "
                "Install it with: pip install pybullet"
            )

        self.config = config or PhysicsConfig()
        self.camera = camera or CameraSpec()
        self._gui = gui
        self._client: int = -1
        self._body_ids: list[int] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to physics server and configure the simulation."""
        mode = pb.GUI if self._gui else pb.DIRECT
        self._client = pb.connect(mode)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        pb.setGravity(0, 0, self.config.gravity, physicsClientId=self._client)
        pb.setTimeStep(self.config.time_step, physicsClientId=self._client)
        pb.setPhysicsEngineParameter(
            numSolverIterations=self.config.solver_iterations,
            physicsClientId=self._client,
        )
        if self.config.use_real_time:
            pb.setRealTimeSimulation(1, physicsClientId=self._client)

    def stop(self) -> None:
        """Disconnect from physics server."""
        if self._client >= 0:
            try:
                pb.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = -1
            self._body_ids.clear()

    def reset(self) -> None:
        """Remove all bodies and reload the ground plane."""
        pb.resetSimulation(physicsClientId=self._client)
        pb.setGravity(0, 0, self.config.gravity, physicsClientId=self._client)
        pb.setTimeStep(self.config.time_step, physicsClientId=self._client)
        self._body_ids.clear()

    def step(self) -> None:
        """Advance the simulation by one time step."""
        pb.stepSimulation(physicsClientId=self._client)

    # ------------------------------------------------------------------
    # Object creation helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> int:
        return self._client

    def load_urdf(
        self,
        path: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        fixed: bool = False,
        scale: float = 1.0,
    ) -> int:
        flags = pb.URDF_USE_SELF_COLLISION
        body_id = pb.loadURDF(
            path,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=fixed,
            globalScaling=scale,
            physicsClientId=self._client,
            flags=flags,
        )
        self._body_ids.append(body_id)
        return body_id

    def create_box(
        self,
        half_extents: Tuple[float, float, float],
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
        lateral_friction: float = 0.5,
        restitution: float = 0.1,
    ) -> int:
        col_id = pb.createCollisionShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client,
        )
        vis_id = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._client,
        )
        body_id = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self._client,
        )
        pb.changeDynamics(
            body_id,
            -1,
            lateralFriction=lateral_friction,
            restitution=restitution,
            physicsClientId=self._client,
        )
        self._body_ids.append(body_id)
        return body_id

    def create_cylinder(
        self,
        radius: float,
        height: float,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
    ) -> int:
        col_id = pb.createCollisionShape(
            pb.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self._client,
        )
        vis_id = pb.createVisualShape(
            pb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=self._client,
        )
        body_id = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self._client,
        )
        self._body_ids.append(body_id)
        return body_id

    def create_sphere(
        self,
        radius: float,
        position: Tuple[float, float, float],
        mass: float = 0.0,
        color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
    ) -> int:
        col_id = pb.createCollisionShape(
            pb.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self._client,
        )
        vis_id = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self._client,
        )
        body_id = pb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            physicsClientId=self._client,
        )
        self._body_ids.append(body_id)
        return body_id

    def remove_body(self, body_id: int) -> None:
        pb.removeBody(body_id, physicsClientId=self._client)
        if body_id in self._body_ids:
            self._body_ids.remove(body_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position_orientation(
        self, body_id: int
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        pos, orn = pb.getBasePositionAndOrientation(
            body_id, physicsClientId=self._client
        )
        return pos, orn

    def get_velocity(
        self, body_id: int
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        lin, ang = pb.getBaseVelocity(body_id, physicsClientId=self._client)
        return lin, ang

    def set_position_orientation(
        self,
        body_id: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> None:
        pb.resetBasePositionAndOrientation(
            body_id, position, orientation, physicsClientId=self._client
        )

    def apply_force(
        self,
        body_id: int,
        force: Tuple[float, float, float],
        position: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if position is None:
            pos, _ = self.get_position_orientation(body_id)
            position = pos
        pb.applyExternalForce(
            body_id,
            -1,
            force,
            position,
            pb.WORLD_FRAME,
            physicsClientId=self._client,
        )

    def apply_torque(
        self, body_id: int, torque: Tuple[float, float, float]
    ) -> None:
        pb.applyExternalTorque(
            body_id,
            -1,
            torque,
            pb.WORLD_FRAME,
            physicsClientId=self._client,
        )

    def get_contact_points(self, body_a: int, body_b: Optional[int] = None):
        if body_b is None:
            return pb.getContactPoints(body_a, physicsClientId=self._client)
        return pb.getContactPoints(body_a, body_b, physicsClientId=self._client)

    def ray_test(
        self,
        from_pos: Tuple[float, float, float],
        to_pos: Tuple[float, float, float],
    ):
        """Single ray-cast; returns (hit_object_id, hit_fraction, hit_pos)."""
        result = pb.rayTest(from_pos, to_pos, physicsClientId=self._client)
        if result:
            hit = result[0]
            return hit[0], hit[2], hit[3]  # object_id, fraction, position
        return -1, 1.0, to_pos

    # ------------------------------------------------------------------
    # Camera / vision rendering
    # ------------------------------------------------------------------

    def render_camera(
        self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        width: Optional[int] = None,
        height: Optional[int] = None,
        fov: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render the scene from the given viewpoint.

        Returns
        -------
        np.ndarray
            RGB image of shape (H, W, 3), dtype uint8.
        """
        w = width or self.camera.width
        h = height or self.camera.height
        f = fov or self.camera.fov

        view_mat = pb.computeViewMatrix(eye, target, up, physicsClientId=self._client)
        proj_mat = pb.computeProjectionMatrixFOV(
            fov=f,
            aspect=w / h,
            nearVal=self.camera.near,
            farVal=self.camera.far,
            physicsClientId=self._client,
        )
        _, _, rgb, _, _ = pb.getCameraImage(
            width=w,
            height=h,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        # rgb is (H*W*4) RGBA as flat array or numpy array
        img = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)
        return img[:, :, :3]  # drop alpha

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def euler_to_quaternion(
        roll: float, pitch: float, yaw: float
    ) -> Tuple[float, float, float, float]:
        if not _PYBULLET_AVAILABLE:
            # Fallback pure-Python implementation
            cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
            cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            return (x, y, z, w)
        return pb.getQuaternionFromEuler([roll, pitch, yaw])

    @staticmethod
    def quaternion_to_euler(
        quaternion: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        if not _PYBULLET_AVAILABLE:
            x, y, z, w = quaternion
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(t0, t1)
            t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
            pitch = math.asin(t2)
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(t3, t4)
            return (roll, pitch, yaw)
        return pb.getEulerFromQuaternion(quaternion)

    @property
    def body_ids(self) -> list[int]:
        return list(self._body_ids)

    def __enter__(self) -> "PhysicsWorld":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
