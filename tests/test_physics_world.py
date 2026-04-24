import pytest
import numpy as np
from unittest.mock import patch

pb = pytest.importorskip("pybullet")

from src.emergent_creativity.environment.physics_world import PhysicsWorld, PhysicsConfig, CameraSpec


@pytest.fixture
def physics_world():
    world = PhysicsWorld(gui=False)
    world.start()
    yield world
    world.stop()


def test_lifecycle_start_stop():
    world = PhysicsWorld(gui=False)
    assert world.client == -1
    world.start()
    assert world.client >= 0
    client_id = world.client
    world.stop()
    assert world.client == -1
    # Check reset
    world.start()
    world.create_box((0.5, 0.5, 0.5), (0, 0, 1))
    assert len(world.body_ids) == 1
    world.reset()
    assert len(world.body_ids) == 0
    world.stop()


def test_object_creation(physics_world):
    box_id = physics_world.create_box((0.5, 0.5, 0.5), (0, 0, 1), mass=1.0)
    cyl_id = physics_world.create_cylinder(0.5, 1.0, (2, 0, 1), mass=1.0)
    sph_id = physics_world.create_sphere(0.5, (-2, 0, 1), mass=1.0)

    body_ids = physics_world.body_ids
    assert box_id in body_ids
    assert cyl_id in body_ids
    assert sph_id in body_ids
    assert len(body_ids) == 3


def test_physics_step(physics_world):
    box_id = physics_world.create_box((0.5, 0.5, 0.5), (0, 0, 5.0), mass=1.0)
    initial_pos, _ = physics_world.get_position_orientation(box_id)

    # Step physics a few times to let gravity act
    for _ in range(10):
        physics_world.step()

    final_pos, _ = physics_world.get_position_orientation(box_id)
    # The z coordinate should decrease due to gravity (-9.81)
    assert final_pos[2] < initial_pos[2]


def test_position_orientation_queries(physics_world):
    box_id = physics_world.create_box((0.5, 0.5, 0.5), (0, 0, 1))

    new_pos = (1.0, 2.0, 3.0)
    new_orn = physics_world.euler_to_quaternion(0, 0, 1.57)

    physics_world.set_position_orientation(box_id, new_pos, new_orn)
    pos, orn = physics_world.get_position_orientation(box_id)

    # Assert close since floats might have minor precision diffs
    np.testing.assert_allclose(pos, new_pos, atol=1e-5)
    np.testing.assert_allclose(orn, new_orn, atol=1e-5)


def test_ray_test(physics_world):
    # Put a large box at origin
    box_id = physics_world.create_box((1.0, 1.0, 1.0), (0, 0, 0))

    # Ray from high Z down to origin
    hit_obj, fraction, pos = physics_world.ray_test((0, 0, 5), (0, 0, -5))

    assert hit_obj == box_id
    assert fraction < 1.0


def test_render_camera(physics_world):
    # Just need to check the shape matches the camera spec
    w, h = 84, 84
    physics_world.camera = CameraSpec(width=w, height=h)

    img = physics_world.render_camera(
        eye=(5, 5, 5),
        target=(0, 0, 0)
    )

    assert isinstance(img, np.ndarray)
    assert img.shape == (h, w, 3)


def test_disconnect_exception():
    world = PhysicsWorld(gui=False)
    # Simulate an active connection without actually starting pybullet
    world._client = 1
    world._body_ids = [999]

    with patch("src.emergent_creativity.environment.physics_world.pb.disconnect") as mock_disconnect:
        mock_disconnect.side_effect = Exception("Simulated disconnect failure")

        # stop() should catch the exception and reset client/body_ids
        world.stop()

    assert world.client == -1
    assert len(world.body_ids) == 0
