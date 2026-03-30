"""
tests/test_agent.py
===================
Unit tests for the tenant agent and vitals.
"""
from __future__ import annotations

import pytest

from src.emergent_creativity.tenant.agent import Vitals


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
