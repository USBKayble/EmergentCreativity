"""
tests/test_objects.py
=====================
Unit tests for the objects module and ObjectRegistry.
No PyBullet required.
"""
from __future__ import annotations

import math
import pytest

from src.emergent_creativity.environment.objects import (
    ObjectCategory,
    ObjectRegistry,
    SmellType,
    TasteType,
    WorldObject,
    make_apple,
    make_pizza,
    make_dirty_sock,
    make_tv,
    make_bed,
    OBJECT_FACTORIES,
)


# ---------------------------------------------------------------------------
# WorldObject factory tests
# ---------------------------------------------------------------------------

class TestObjectFactories:
    def test_apple_properties(self):
        apple = make_apple()
        assert apple.name == "apple"
        assert apple.is_food
        assert apple.can_pick_up
        assert apple.nutrition > 0
        assert apple.sensory.smell_type == SmellType.FOOD
        assert apple.sensory.taste_type == TasteType.SWEET

    def test_pizza_high_nutrition(self):
        pizza = make_pizza()
        assert pizza.nutrition >= 0.5

    def test_dirty_sock_is_mess(self):
        sock = make_dirty_sock()
        assert sock.is_mess
        assert sock.on_floor
        assert sock.sensory.smell_type == SmellType.GARBAGE

    def test_tv_not_pickupable(self):
        tv = make_tv()
        assert not tv.can_pick_up
        assert tv.is_interactive
        assert tv.sensory.sound_level > 0

    def test_bed_not_pickupable_and_interactive(self):
        bed = make_bed()
        assert not bed.can_pick_up
        assert bed.is_interactive
        assert bed.is_surface
        assert bed.energy_restore > 0

    def test_all_factories_callable(self):
        for name, factory in OBJECT_FACTORIES.items():
            obj = factory()
            assert isinstance(obj, WorldObject), f"{name} factory did not return WorldObject"
            assert obj.name == name or name in obj.name


# ---------------------------------------------------------------------------
# ObjectRegistry tests
# ---------------------------------------------------------------------------

class TestObjectRegistry:
    def _make_obj(self, body_id: int, name: str = "test") -> WorldObject:
        obj = WorldObject(
            name=name,
            category=ObjectCategory.MISC,
            body_id=body_id,
        )
        return obj

    def test_register_and_get(self):
        reg = ObjectRegistry()
        obj = self._make_obj(1, "box")
        reg.register(obj)
        assert reg.get(1) is obj

    def test_register_invalid_body_raises(self):
        reg = ObjectRegistry()
        obj = WorldObject(name="bad", category=ObjectCategory.MISC, body_id=-1)
        with pytest.raises(ValueError):
            reg.register(obj)

    def test_unregister(self):
        reg = ObjectRegistry()
        obj = self._make_obj(2, "removable")
        reg.register(obj)
        reg.unregister(2)
        assert reg.get(2) is None

    def test_distance_calculation(self):
        reg = ObjectRegistry()
        obj = self._make_obj(3, "far_obj")
        reg.register(obj)
        reg.update_position(3, (3.0, 4.0, 0.0))
        dist = reg.distance(3, (0.0, 0.0, 0.0))
        assert math.isclose(dist, 5.0, rel_tol=1e-5)

    def test_distance_unknown_object(self):
        reg = ObjectRegistry()
        assert reg.distance(999, (0, 0, 0)) == float("inf")

    def test_objects_within_radius(self):
        reg = ObjectRegistry()
        for i in range(5):
            obj = self._make_obj(i + 10, f"obj_{i}")
            reg.register(obj)
            reg.update_position(i + 10, (float(i), 0.0, 0.0))

        nearby = reg.objects_within_radius((0.0, 0.0, 0.0), radius=2.5)
        ids = {o.body_id for o in nearby}
        assert 10 in ids   # (0,0)
        assert 11 in ids   # (1,0)
        assert 12 in ids   # (2,0)
        assert 13 not in ids   # (3,0) > 2.5

    def test_nearest_object(self):
        reg = ObjectRegistry()
        for i in range(3):
            obj = self._make_obj(i + 20, f"n_{i}")
            reg.register(obj)
            reg.update_position(i + 20, (float(i * 5), 0.0, 0.0))
        nearest = reg.nearest((0.0, 0.0, 0.0))
        assert nearest.body_id == 20  # closest to origin

    def test_mess_count(self):
        reg = ObjectRegistry()
        sock = make_dirty_sock()
        sock.body_id = 100
        sock.on_floor = True
        reg.register(sock)

        paper = make_dirty_sock()
        paper.name = "paper"
        paper.body_id = 101
        paper.on_floor = False  # tidied up
        reg.register(paper)

        assert reg.mess_count() == 1

    def test_all_returns_list(self):
        reg = ObjectRegistry()
        for i in range(3):
            reg.register(self._make_obj(i + 30))
        assert len(reg.all()) == 3

    def test_clear(self):
        reg = ObjectRegistry()
        reg.register(self._make_obj(40))
        reg.clear()
        assert reg.all() == []
