"""
tests/test_senses.py
====================
Unit tests for the sensory system.
Requires only numpy; no PyBullet needed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.emergent_creativity.environment.objects import (
    ObjectCategory,
    ObjectRegistry,
    SmellType,
    TasteType,
    WorldObject,
    make_apple,
    make_tv,
    make_dirty_sock,
    SensoryProfile,
)
from src.emergent_creativity.environment.senses import (
    HEARING_DIM,
    SMELL_DIM,
    TASTE_DIM,
    TOUCH_DIM,
    VISION_H,
    VISION_W,
    VISION_C,
    HearingSense,
    SmellSense,
    TasteSense,
    SensoryObservation,
    TOTAL_SENSORY_DIM,
)


# ---------------------------------------------------------------------------
# HearingSense tests
# ---------------------------------------------------------------------------

class TestHearingSense:
    def _make_registry_with_tv(self, tv_pos=(2.0, 0.0, 0.0)):
        reg = ObjectRegistry()
        tv = make_tv()
        tv.body_id = 1
        tv.active = True
        reg.register(tv)
        reg.update_position(1, tv_pos)
        return reg

    def test_tv_audible_nearby(self):
        reg = self._make_registry_with_tv(tv_pos=(1.0, 0.0, 0.0))
        sense = HearingSense(reg)
        levels = sense.observe((0.0, 0.0, 0.0))
        assert levels.shape == (HEARING_DIM,)
        tv_idx = HearingSense.CHANNELS.index("tv_audio")
        assert levels[tv_idx] > 0.0

    def test_tv_silent_when_far_away(self):
        reg = self._make_registry_with_tv(tv_pos=(100.0, 0.0, 0.0))
        sense = HearingSense(reg)
        levels = sense.observe((0.0, 0.0, 0.0))
        tv_idx = HearingSense.CHANNELS.index("tv_audio")
        assert levels[tv_idx] == 0.0

    def test_closer_is_louder(self):
        reg_near = self._make_registry_with_tv(tv_pos=(1.0, 0.0, 0.0))
        reg_far  = self._make_registry_with_tv(tv_pos=(5.0, 0.0, 0.0))
        sense_near = HearingSense(reg_near)
        sense_far  = HearingSense(reg_far)
        tv_idx = HearingSense.CHANNELS.index("tv_audio")
        near_lvl = sense_near.observe((0.0, 0.0, 0.0))[tv_idx]
        far_lvl  = sense_far.observe((0.0, 0.0, 0.0))[tv_idx]
        assert near_lvl > far_lvl

    def test_values_in_range(self):
        reg = self._make_registry_with_tv()
        sense = HearingSense(reg)
        levels = sense.observe((0.0, 0.0, 0.0))
        assert np.all(levels >= 0.0)
        assert np.all(levels <= 1.0)

    def test_empty_registry_all_silent(self):
        reg = ObjectRegistry()
        sense = HearingSense(reg)
        levels = sense.observe((0.0, 0.0, 0.0))
        assert np.all(levels == 0.0)


# ---------------------------------------------------------------------------
# SmellSense tests
# ---------------------------------------------------------------------------

class TestSmellSense:
    def _make_reg(self, smell_type, intensity, position):
        reg = ObjectRegistry()
        obj = WorldObject(
            name="smelly",
            category=ObjectCategory.MISC,
            body_id=1,
            sensory=SensoryProfile(
                smell_type=smell_type,
                smell_intensity=intensity,
            ),
        )
        reg.register(obj)
        reg.update_position(1, position)
        return reg

    def test_smell_detected_nearby(self):
        reg = self._make_reg(SmellType.FOOD, 0.8, (1.0, 0.0, 0.0))
        sense = SmellSense(reg, radius=3.0)
        levels = sense.observe((0.0, 0.0, 0.0))
        food_idx = list(SmellType).index(SmellType.FOOD)
        assert levels[food_idx] > 0.0

    def test_smell_zero_when_out_of_radius(self):
        reg = self._make_reg(SmellType.FOOD, 0.8, (10.0, 0.0, 0.0))
        sense = SmellSense(reg, radius=3.0)
        levels = sense.observe((0.0, 0.0, 0.0))
        assert np.all(levels == 0.0)

    def test_smell_decays_with_distance(self):
        reg_near = self._make_reg(SmellType.GARBAGE, 1.0, (0.5, 0.0, 0.0))
        reg_far  = self._make_reg(SmellType.GARBAGE, 1.0, (2.5, 0.0, 0.0))
        sense_near = SmellSense(reg_near, radius=3.0)
        sense_far  = SmellSense(reg_far,  radius=3.0)
        idx = list(SmellType).index(SmellType.GARBAGE)
        assert sense_near.observe((0, 0, 0))[idx] > sense_far.observe((0, 0, 0))[idx]

    def test_output_shape(self):
        reg = ObjectRegistry()
        sense = SmellSense(reg)
        levels = sense.observe((0.0, 0.0, 0.0))
        assert levels.shape == (SMELL_DIM,)

    def test_values_in_range(self):
        reg = self._make_reg(SmellType.FOOD, 1.0, (0.5, 0.0, 0.0))
        sense = SmellSense(reg, radius=3.0)
        levels = sense.observe((0.0, 0.0, 0.0))
        assert np.all(levels >= 0.0)
        assert np.all(levels <= 1.0)


# ---------------------------------------------------------------------------
# TasteSense tests
# ---------------------------------------------------------------------------

class TestTasteSense:
    def test_no_taste_before_eating(self):
        ts = TasteSense()
        result = ts.observe()
        assert np.all(result == 0.0)
        assert result.shape == (TASTE_DIM,)

    def test_taste_activated_by_apple(self):
        ts = TasteSense()
        apple = make_apple()
        ts.activate(apple)
        result = ts.observe()
        sweet_idx = list(TasteType).index(TasteType.SWEET)
        assert result[sweet_idx] > 0.0

    def test_taste_fades_over_time(self):
        ts = TasteSense()
        apple = make_apple()
        ts.activate(apple)
        sweet_idx = list(TasteType).index(TasteType.SWEET)
        first = ts.observe()[sweet_idx]
        for _ in range(10):
            val = ts.observe()[sweet_idx]
        # Should have faded
        assert val < first

    def test_taste_gone_after_fade(self):
        ts = TasteSense()
        apple = make_apple()
        ts.activate(apple)
        for _ in range(TasteSense.FADE_STEPS + 2):
            ts.observe()
        assert np.all(ts.observe() == 0.0)

    def test_reset_clears_taste(self):
        ts = TasteSense()
        ts.activate(make_apple())
        ts.reset()
        assert np.all(ts.observe() == 0.0)


# ---------------------------------------------------------------------------
# SensoryObservation tests
# ---------------------------------------------------------------------------

class TestSensoryObservation:
    def test_default_shapes(self):
        obs = SensoryObservation()
        assert obs.vision.shape  == (VISION_H, VISION_W, VISION_C)
        assert obs.hearing.shape == (HEARING_DIM,)
        assert obs.touch.shape   == (TOUCH_DIM,)
        assert obs.smell.shape   == (SMELL_DIM,)
        assert obs.taste.shape   == (TASTE_DIM,)

    def test_to_flat_non_visual_length(self):
        obs = SensoryObservation()
        flat = obs.to_flat_non_visual()
        expected = HEARING_DIM + TOUCH_DIM + SMELL_DIM + TASTE_DIM
        assert flat.shape == (expected,)
        assert flat.shape == (TOTAL_SENSORY_DIM,)

    def test_default_zeros(self):
        obs = SensoryObservation()
        assert np.all(obs.vision  == 0.0)
        assert np.all(obs.hearing == 0.0)
        assert np.all(obs.touch   == 0.0)
        assert np.all(obs.smell   == 0.0)
        assert np.all(obs.taste   == 0.0)
