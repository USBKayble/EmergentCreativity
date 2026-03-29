"""
tests/test_actions.py
=====================
Unit tests for the Action enum and action-space definitions.
No heavy dependencies required.
"""
from __future__ import annotations

import pytest

from src.emergent_creativity.tenant.actions import (
    Action,
    ACTION_LABELS,
    N_ACTIONS,
    MOVE_SPEED,
    TURN_SPEED_DEG,
    REACH_DISTANCE,
)


class TestActionEnum:
    def test_n_actions_matches_enum(self):
        assert N_ACTIONS == len(Action)

    def test_all_actions_have_labels(self):
        for act in Action:
            assert int(act) in ACTION_LABELS, f"Action {act} missing label"

    def test_action_values_are_sequential(self):
        values = sorted(int(a) for a in Action)
        expected = list(range(N_ACTIONS))
        assert values == expected

    def test_action_names(self):
        # Spot-check key actions exist
        assert Action.IDLE          == 0
        assert Action.MOVE_FORWARD  == 1
        assert Action.MOVE_BACKWARD == 2
        assert Action.PICK_UP       == 7
        assert Action.EAT           == 10
        assert Action.SLEEP         == 11
        assert Action.USE_BATHROOM  == 12

    def test_move_speed_positive(self):
        assert MOVE_SPEED > 0

    def test_turn_speed_positive(self):
        assert TURN_SPEED_DEG > 0

    def test_reach_distance_positive(self):
        assert REACH_DISTANCE > 0

    def test_label_for_idle(self):
        assert ACTION_LABELS[Action.IDLE] == "Idle"

    def test_label_for_eat(self):
        assert "Eat" in ACTION_LABELS[Action.EAT]
