"""
tests/test_rewards.py
=====================
Unit tests for the reward ruleset system.
These tests do NOT require PyBullet, Torch, or any heavy dependency.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _MockVitals:
    def __init__(self, hunger=0.2, energy=0.8, bladder=0.1, happiness=0.6):
        self.hunger = hunger
        self.energy = energy
        self.bladder = bladder
        self.happiness = happiness


class _MockRegistry:
    def __init__(self, mess_count=0):
        self._mess = mess_count

    def mess_count(self):
        return self._mess

    def all(self):
        return []

    def distance(self, body_id, pos):
        return 10.0


class _MockTenant:
    def __init__(
        self,
        hunger=0.2,
        energy=0.8,
        bladder=0.1,
        happiness=0.6,
        mess_count=0,
        events=None,
        is_sleeping=False,
        is_watching_tv=False,
        is_reading=False,
        is_playing_game=False,
        idle_steps=0,
        total_steps=0,
    ):
        self.vitals = _MockVitals(hunger, energy, bladder, happiness)
        self.events = list(events or [])
        self.is_sleeping = is_sleeping
        self.is_watching_tv = is_watching_tv
        self.is_reading = is_reading
        self.is_playing_game = is_playing_game
        self.idle_steps = idle_steps
        self.total_steps = total_steps
        self._mess_count = mess_count

    @property
    def registry(self):
        return _MockRegistry(self._mess_count)

    def get_position(self):
        return (2.5, 2.5, 0.0)


# ---------------------------------------------------------------------------
# Tests for Rule parsing
# ---------------------------------------------------------------------------

from src.emergent_creativity.rewards.ruleset import Rule, RewardEvaluator


class TestRuleParsing:
    def test_event_rule(self):
        rule = Rule({"name": "eat_food", "event": "ate_food", "reward": 10.0})
        assert rule.name == "eat_food"
        assert rule.event == "ate_food"
        assert rule.reward == 10.0
        assert not rule.per_step

    def test_per_step_rule(self):
        rule = Rule(
            {
                "name": "hungry_penalty",
                "per_step": True,
                "condition": "hunger > 0.6",
                "reward": -0.5,
            }
        )
        assert rule.per_step
        assert rule.condition == "hunger > 0.6"

    def test_condition_true(self):
        rule = Rule({"condition": "hunger > 0.6", "reward": -1.0})
        ctx = {
            "hunger": 0.8,
            "energy": 0.5,
            "bladder": 0.0,
            "happiness": 0.5,
            "mess_count": 0,
            "idle_steps": 0,
            "is_sleeping": False,
            "is_watching_tv": False,
            "is_reading": False,
            "is_playing_game": False,
        }
        assert rule.evaluate_condition(ctx) is True

    def test_condition_false(self):
        rule = Rule({"condition": "hunger > 0.6", "reward": -1.0})
        ctx = {
            "hunger": 0.2,
            "energy": 0.5,
            "bladder": 0.0,
            "happiness": 0.5,
            "mess_count": 0,
            "idle_steps": 0,
            "is_sleeping": False,
            "is_watching_tv": False,
            "is_reading": False,
            "is_playing_game": False,
        }
        assert rule.evaluate_condition(ctx) is False

    def test_no_condition_always_true(self):
        rule = Rule({"reward": 1.0})
        assert rule.evaluate_condition({}) is True

    def test_matches_event(self):
        rule = Rule({"event": "ate_food", "reward": 10.0})
        assert rule.matches_event("ate_food") is True
        assert rule.matches_event("picked_up_mess") is False

    def test_invalid_condition_returns_false(self):
        rule = Rule({"condition": "undefined_var > 0", "reward": 1.0})
        assert rule.evaluate_condition({}) is False


# ---------------------------------------------------------------------------
# Tests for RewardEvaluator
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "rules": [
        {"name": "eat_food", "event": "ate_food", "reward": 10.0},
        {"name": "pick_up_mess", "event": "picked_up_mess", "reward": 5.0},
        {
            "name": "hungry_penalty",
            "per_step": True,
            "condition": "hunger > 0.6",
            "reward": -0.5,
        },
        {
            "name": "mess_penalty",
            "per_step": True,
            "condition": "mess_count > 0",
            "reward": -0.1,
        },
        {
            "name": "idle_penalty",
            "per_step": True,
            "condition": "idle_steps > 100",
            "reward": -0.2,
        },
    ],
    "vitals": {},
    "terminal": {
        "hunger_max": 1.0,
        "energy_min": 0.0,
        "max_steps": 1000,
    },
}


class TestRewardEvaluator:
    def _evaluator(self):
        return RewardEvaluator.from_dict(MINIMAL_CONFIG)

    def test_eat_event_reward(self):
        ev = self._evaluator()
        tenant = _MockTenant(events=["ate_food"])
        registry = _MockRegistry(mess_count=0)
        total, info = ev.evaluate(tenant, registry)
        assert total == pytest.approx(10.0)
        assert "eat_food" in info

    def test_pick_up_mess_event(self):
        ev = self._evaluator()
        tenant = _MockTenant(events=["picked_up_mess"])
        registry = _MockRegistry(mess_count=0)
        total, info = ev.evaluate(tenant, registry)
        assert total == pytest.approx(5.0)

    def test_no_reward_for_idle_tenant(self):
        ev = self._evaluator()
        tenant = _MockTenant()
        registry = _MockRegistry()
        total, _ = ev.evaluate(tenant, registry)
        assert total == pytest.approx(0.0)

    def test_hungry_per_step_penalty(self):
        ev = self._evaluator()
        tenant = _MockTenant(hunger=0.8)
        registry = _MockRegistry()
        total, info = ev.evaluate(tenant, registry)
        assert "hungry_penalty" in info
        assert total <= -0.5

    def test_mess_per_step_penalty_scales_with_count(self):
        ev = self._evaluator()
        tenant = _MockTenant()
        registry1 = _MockRegistry(mess_count=1)
        registry3 = _MockRegistry(mess_count=3)
        _, info1 = ev.evaluate(tenant, registry1)
        _, info3 = ev.evaluate(tenant, registry3)
        # More mess → more penalty
        assert abs(info3.get("mess_penalty", 0.0)) > abs(info1.get("mess_penalty", 0.0))

    def test_combined_event_and_per_step(self):
        ev = self._evaluator()
        tenant = _MockTenant(hunger=0.8, events=["ate_food"])
        registry = _MockRegistry()
        total, info = ev.evaluate(tenant, registry)
        # +10 eat, -0.5 hunger penalty
        assert "eat_food" in info
        assert "hungry_penalty" in info
        assert total == pytest.approx(10.0 - 0.5)

    def test_idle_penalty(self):
        ev = self._evaluator()
        tenant = _MockTenant(idle_steps=200)
        registry = _MockRegistry()
        total, info = ev.evaluate(tenant, registry)
        assert "idle_penalty" in info
        assert total < 0.0

    # --- Terminal conditions ---

    def test_terminal_on_starvation(self):
        ev = self._evaluator()
        tenant = _MockTenant(hunger=1.0)
        assert ev.is_terminal(tenant) is True

    def test_terminal_on_exhaustion(self):
        ev = self._evaluator()
        tenant = _MockTenant(energy=0.0)
        assert ev.is_terminal(tenant) is True

    def test_terminal_on_max_steps(self):
        ev = self._evaluator()
        tenant = _MockTenant(total_steps=1000)
        assert ev.is_terminal(tenant) is True

    def test_not_terminal_during_normal_play(self):
        ev = self._evaluator()
        tenant = _MockTenant(hunger=0.3, energy=0.7, total_steps=50)
        assert ev.is_terminal(tenant) is False

    # --- YAML loading ---

    def test_from_yaml_loads_rules(self):
        import os

        cfg = os.path.join(os.path.dirname(__file__), "..", "config", "rewards.yaml")
        ev = RewardEvaluator.from_yaml(cfg)
        assert len(ev.rules) > 0

    def test_from_yaml_has_terminal_config(self):
        import os

        cfg = os.path.join(os.path.dirname(__file__), "..", "config", "rewards.yaml")
        ev = RewardEvaluator.from_yaml(cfg)
        assert ev.max_steps > 0
