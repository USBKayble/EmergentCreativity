"""
tests/test_apartment.py
=======================
Unit tests for the apartment builder.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from src.emergent_creativity.environment.apartment import Apartment

class TestApartment:
    def test_build(self):
        """
        Test that Apartment.build() calls the correct internal methods to build
        the apartment layout from scratch.
        """
        world_mock = Mock()
        registry_mock = Mock()

        apartment = Apartment(world=world_mock, registry=registry_mock)

        with patch.object(apartment, '_build_floor') as mock_build_floor, \
             patch.object(apartment, '_build_walls') as mock_build_walls, \
             patch.object(apartment, '_build_furniture') as mock_build_furniture, \
             patch.object(apartment, '_place_items') as mock_place_items:

            apartment.build()

            mock_build_floor.assert_called_once()
            mock_build_walls.assert_called_once()
            mock_build_furniture.assert_called_once()
            mock_place_items.assert_called_once()
