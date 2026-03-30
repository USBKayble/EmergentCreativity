"""
tests/test_apartment.py
=======================
Unit tests for the Apartment class.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.emergent_creativity.environment.apartment import Apartment
from src.emergent_creativity.environment.objects import (
    ObjectRegistry,
    WorldObject,
)
from src.emergent_creativity.environment.physics_world import PhysicsWorld


class TestApartmentInit:
    def test_initialization(self):
        world_mock = MagicMock(spec=PhysicsWorld)
        registry_mock = MagicMock(spec=ObjectRegistry)

        apartment = Apartment(
            world=world_mock, registry=registry_mock, seed=123
        )

        assert apartment._world is world_mock
        assert apartment._registry is registry_mock
        assert apartment.furniture == []
        assert apartment.items == []
        assert apartment._static_ids == []


class TestApartmentBuild:
    def test_build_calls_internal_methods(self):
        world_mock = MagicMock(spec=PhysicsWorld)
        registry_mock = MagicMock(spec=ObjectRegistry)

        # We need world_mock.create_box to return sequential IDs
        # to avoid conflicts or errors
        world_mock.create_box.side_effect = range(1, 1000)

        apartment = Apartment(world=world_mock, registry=registry_mock)

        # Call build
        apartment.build()

        # Verify that create_box was called multiple times
        assert world_mock.create_box.call_count > 0

        # Verify furniture and items were populated
        assert len(apartment.furniture) > 0
        assert len(apartment.items) > 0
        assert len(apartment._static_ids) > 0

        # Verify objects were registered in the registry
        total_items = len(apartment.furniture) + len(apartment.items)
        assert registry_mock.register.call_count == total_items
        assert registry_mock.update_position.call_count == total_items


class TestApartmentResetItems:
    def test_reset_items(self):
        world_mock = MagicMock(spec=PhysicsWorld)
        registry_mock = MagicMock(spec=ObjectRegistry)

        # Setup create_box to return unique IDs
        world_mock.create_box.side_effect = range(1, 1000)

        apartment = Apartment(world=world_mock, registry=registry_mock)

        # Mocking an existing item
        dummy_item = MagicMock(spec=WorldObject)
        dummy_item.body_id = 42
        # Mock attributes because dataclasses or eq might access them
        dummy_item.name = "dummy_item"
        dummy_item.category = None
        dummy_item.sensory = None
        dummy_item.color = None
        dummy_item.half_extents = None
        dummy_item.mass = 0.0
        apartment._items.append(dummy_item)

        apartment.reset_items()

        # Check that old item was removed from world and registry
        world_mock.remove_body.assert_called_with(42)
        registry_mock.unregister.assert_called_with(42)

        # Ensure new items are generated (at least one)
        assert len(apartment.items) > 0

        # Check that the old item is no longer in the items list
        assert dummy_item not in apartment.items


class TestApartmentSyncRegistry:
    def test_sync_registry_updates_positions(self):
        world_mock = MagicMock(spec=PhysicsWorld)
        registry_mock = MagicMock(spec=ObjectRegistry)

        apartment = Apartment(world=world_mock, registry=registry_mock)

        # Add a dummy item
        dummy_item = MagicMock(spec=WorldObject)
        dummy_item.body_id = 10
        dummy_item.consumed = False
        dummy_item.can_pick_up = True
        dummy_item.held_by_agent = False
        apartment._items.append(dummy_item)

        # Add a dummy furniture
        dummy_furniture = MagicMock(spec=WorldObject)
        dummy_furniture.body_id = 20
        dummy_furniture.can_pick_up = False
        dummy_furniture.held_by_agent = False
        apartment._furniture.append(dummy_furniture)

        # Mock get_position_orientation to return a dummy pos and orn
        world_mock.get_position_orientation.return_value = (
            (1.0, 2.0, 0.1),
            (0.0, 0.0, 0.0, 1.0),
        )

        apartment.sync_registry()

        # Verify registry was updated with new positions
        registry_mock.update_position.assert_any_call(10, (1.0, 2.0, 0.1))
        registry_mock.update_position.assert_any_call(20, (1.0, 2.0, 0.1))

        # Verify on_floor property was updated for the item (z < 0.15)
        assert dummy_item.on_floor is True

        # Test item consumption removes it
        dummy_item.consumed = True
        apartment.sync_registry()
        assert dummy_item not in apartment.items

    def test_sync_registry_ignores_negative_body_id(self):
        world_mock = MagicMock(spec=PhysicsWorld)
        registry_mock = MagicMock(spec=ObjectRegistry)

        apartment = Apartment(world=world_mock, registry=registry_mock)

        dummy_item = MagicMock(spec=WorldObject)
        dummy_item.body_id = -1
        apartment._items.append(dummy_item)

        dummy_furniture = MagicMock(spec=WorldObject)
        dummy_furniture.body_id = -1
        apartment._furniture.append(dummy_furniture)

        apartment.sync_registry()

        # update_position shouldn't be called
        registry_mock.update_position.assert_not_called()
