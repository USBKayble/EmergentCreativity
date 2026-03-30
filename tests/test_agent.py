"""
tests/test_agent.py
===================
Unit tests for the tenant agent and vitals.
"""
from __future__ import annotations

import pytest

from unittest.mock import MagicMock

from src.emergent_creativity.tenant.agent import Tenant, Vitals
from src.emergent_creativity.tenant.actions import Action
from src.emergent_creativity.environment.physics_world import PhysicsWorld
from src.emergent_creativity.environment.objects import ObjectRegistry
from src.emergent_creativity.environment.senses import SensorySuite


class TestVitals:
    def test_clamp_below_zero(self):
        v = Vitals(hunger=-0.5, energy=-1.0, bladder=-0.1, happiness=-0.99)
        v.clamp()
        assert v.hunger == 0.0
        assert v.energy == 0.0
        assert v.bladder == 0.0
        assert v.happiness == 0.0

    def test_clamp_above_one(self):
        v = Vitals(hunger=1.5, energy=2.0, bladder=1.01, happiness=10.0)
        v.clamp()
        assert v.hunger == 1.0
        assert v.energy == 1.0
        assert v.bladder == 1.0
        assert v.happiness == 1.0

    def test_clamp_exact_boundaries(self):
        v = Vitals(hunger=0.0, energy=1.0, bladder=0.0, happiness=1.0)
        v.clamp()
        assert v.hunger == 0.0
        assert v.energy == 1.0
        assert v.bladder == 0.0
        assert v.happiness == 1.0

    def test_clamp_within_range(self):
        v = Vitals(hunger=0.5, energy=0.75, bladder=0.2, happiness=0.9)
        v.clamp()
        assert v.hunger == 0.5
        assert v.energy == 0.75
        assert v.bladder == 0.2
        assert v.happiness == 0.9


class TestTenantStep:
    @pytest.fixture
    def mock_tenant(self):
        world = MagicMock(spec=PhysicsWorld)
        registry = MagicMock(spec=ObjectRegistry)
        sensors = MagicMock(spec=SensorySuite)
        tenant = Tenant(world, registry, sensors)
        # Set body_id to a valid value by default
        tenant._body_id = 1
        return tenant

    def test_step_early_return_when_no_body(self, mock_tenant):
        # We need to explicitly patch _move because the real step calls it.
        # However, early return should skip it.
        mock_tenant._body_id = -1
        mock_tenant._move = MagicMock()
        mock_tenant.step(Action.MOVE_FORWARD.value)
        mock_tenant._move.assert_not_called()

    def test_step_idle(self, mock_tenant):
        mock_tenant._move = MagicMock()  # Mock _move to prevent it from calling into world mock
        mock_tenant._update_vitals = MagicMock()
        mock_tenant.step(Action.IDLE.value)
        assert mock_tenant.idle_steps == 1
        mock_tenant.step(Action.MOVE_FORWARD.value)
        assert mock_tenant.idle_steps == 0

    @pytest.mark.parametrize("action, method_name", [
        (Action.MOVE_FORWARD, "_move"),
        (Action.MOVE_BACKWARD, "_move"),
        (Action.MOVE_LEFT, "_move"),
        (Action.MOVE_RIGHT, "_move"),
        (Action.PICK_UP, "_pick_up"),
        (Action.PUT_DOWN, "_put_down"),
        (Action.INTERACT, "_interact"),
        (Action.EAT, "_eat"),
        (Action.SLEEP, "_sleep"),
        (Action.USE_BATHROOM, "_use_bathroom"),
    ])
    def test_step_action_dispatch(self, mock_tenant, action, method_name):
        method_mock = MagicMock()
        setattr(mock_tenant, method_name, method_mock)

        # Override _update_vitals so it doesn't interfere
        mock_tenant._update_vitals = MagicMock()

        mock_tenant.step(action.value)

        if method_name == "_move":
            method_mock.assert_called_once_with(action)
        else:
            method_mock.assert_called_once()
