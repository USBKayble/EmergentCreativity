import pytest
import os
import tempfile
import pybullet as pb
from src.emergent_creativity.environment.physics_world import PhysicsWorld

@pytest.fixture
def physics_world():
    world = PhysicsWorld(gui=False)
    world.start()
    yield world
    world.stop()

@pytest.fixture
def temp_urdf():
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode='w') as f:
        f.write('''<?xml version="1.0"?>
<robot name="simple_box">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
''')
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.remove(filepath)

def test_load_urdf(physics_world, temp_urdf):
    position = (1.0, 2.0, 3.0)
    orientation = physics_world.euler_to_quaternion(0, 0, 0)

    # We use positional arguments to be safe across different potential signatures
    # (The issue description mentioned different kwarg names vs actual code)
    body_id = physics_world.load_urdf(
        temp_urdf,
        position,
        orientation,
        True,
        2.0
    )

    assert body_id >= 0
    assert body_id in physics_world.body_ids

    pos, orn = physics_world.get_position_orientation(body_id)
    assert pos[0] == pytest.approx(1.0)
    assert pos[1] == pytest.approx(2.0)
    assert pos[2] == pytest.approx(3.0)
