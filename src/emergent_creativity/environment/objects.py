"""
objects.py
==========
Definitions for every interactive object that can exist in the apartment.

Each :class:`WorldObject` wraps a PyBullet body ID with gameplay metadata:
* physical properties  (mass, size, colour)
* sensory properties   (smell intensity/type, taste, sound)
* interaction flags    (can_pick_up, is_food, is_mess, is_appliance …)
* mutable state        (on_floor, dirty, active, consumed …)

The :class:`ObjectRegistry` acts as an in-memory catalogue of all live objects
and provides spatial queries (nearest object, objects in radius, etc.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ObjectCategory(Enum):
    FURNITURE    = auto()
    FOOD         = auto()
    BEVERAGE     = auto()
    MESS         = auto()
    APPLIANCE    = auto()
    ENTERTAINMENT = auto()
    CLEANING     = auto()
    HYGIENE      = auto()
    CLOTHING     = auto()
    MISC         = auto()


class SmellType(Enum):
    NONE     = auto()
    FOOD     = auto()
    GARBAGE  = auto()
    CLEAN    = auto()
    MUSTY    = auto()
    FLORAL   = auto()
    CHEMICAL = auto()


class TasteType(Enum):
    NONE    = auto()
    SWEET   = auto()
    SALTY   = auto()
    SOUR    = auto()
    BITTER  = auto()
    UMAMI   = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SensoryProfile:
    """Sensory attributes of an object."""
    smell_type: SmellType = SmellType.NONE
    smell_intensity: float = 0.0          # 0–1
    taste_type: TasteType = TasteType.NONE
    taste_intensity: float = 0.0          # 0–1
    sound_level: float = 0.0              # dB equivalent, 0–100
    sound_label: str = ""                 # e.g. "tv_audio", "running_water"
    temperature: float = 20.0            # Celsius, affects comfort
    texture: float = 0.5                  # 0=smooth, 1=rough (touch)


@dataclass
class WorldObject:
    """
    A single object instance living in the physics world.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "apple", "dirty_sock").
    category : ObjectCategory
    body_id : int
        PyBullet body identifier. -1 when not yet spawned.
    half_extents : tuple
        Approximate bounding box half-extents (x, y, z) in metres.
    mass : float
        kg; 0 = immovable static body.
    color : tuple
        RGBA [0,1].
    sensory : SensoryProfile
    can_pick_up : bool
    is_food : bool
    is_mess : bool
        True when the object is considered "clutter on the floor".
    is_appliance : bool
    is_interactive : bool
        Can the tenant "use" this object (TV, book, fridge …)?
    nutrition : float
        Hunger reduction when eaten (0–1).
    """
    name: str
    category: ObjectCategory
    body_id: int = -1
    half_extents: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    mass: float = 0.5
    color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    sensory: SensoryProfile = field(default_factory=SensoryProfile)

    # Interaction flags
    can_pick_up: bool = True
    is_food: bool = False
    is_mess: bool = False
    is_appliance: bool = False
    is_interactive: bool = False
    is_surface: bool = False    # table-top / shelf etc.

    # Mutable state
    on_floor: bool = False
    dirty: bool = False
    active: bool = False        # e.g. TV turned on
    consumed: bool = False      # food has been eaten
    held_by_agent: bool = False

    # Nutrition / utility values
    nutrition: float = 0.0      # hunger reduction 0–1
    entertainment: float = 0.0  # entertainment value per step 0–1
    energy_restore: float = 0.0 # energy restored if used

    def distance_to(self, other_pos: Tuple[float, float, float]) -> float:
        """Euclidean distance from this object's body_id to a world position."""
        raise NotImplementedError(
            "Call ObjectRegistry.distance_to() which knows object positions."
        )


# ---------------------------------------------------------------------------
# Object factory – pre-defined archetypes
# ---------------------------------------------------------------------------

def make_apple(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="apple",
        category=ObjectCategory.FOOD,
        body_id=body_id,
        half_extents=(0.04, 0.04, 0.04),
        mass=0.18,
        color=(0.9, 0.1, 0.1, 1.0),
        sensory=SensoryProfile(
            smell_type=SmellType.FOOD,
            smell_intensity=0.4,
            taste_type=TasteType.SWEET,
            taste_intensity=0.7,
            texture=0.3,
            temperature=15.0,
        ),
        can_pick_up=True,
        is_food=True,
        nutrition=0.3,
    )


def make_pizza(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="pizza",
        category=ObjectCategory.FOOD,
        body_id=body_id,
        half_extents=(0.15, 0.15, 0.02),
        mass=0.5,
        color=(0.9, 0.6, 0.2, 1.0),
        sensory=SensoryProfile(
            smell_type=SmellType.FOOD,
            smell_intensity=0.85,
            taste_type=TasteType.UMAMI,
            taste_intensity=0.9,
            texture=0.5,
            temperature=60.0,
        ),
        can_pick_up=True,
        is_food=True,
        nutrition=0.7,
    )


def make_water_bottle(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="water_bottle",
        category=ObjectCategory.BEVERAGE,
        body_id=body_id,
        half_extents=(0.035, 0.035, 0.12),
        mass=0.5,
        color=(0.6, 0.85, 1.0, 0.7),
        sensory=SensoryProfile(
            smell_type=SmellType.NONE,
            taste_type=TasteType.NONE,
            texture=0.1,
            temperature=10.0,
        ),
        can_pick_up=True,
        is_food=True,
        nutrition=0.1,
    )


def make_dirty_sock(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="dirty_sock",
        category=ObjectCategory.MESS,
        body_id=body_id,
        half_extents=(0.06, 0.12, 0.01),
        mass=0.05,
        color=(0.6, 0.55, 0.4, 1.0),
        sensory=SensoryProfile(
            smell_type=SmellType.GARBAGE,
            smell_intensity=0.6,
            taste_type=TasteType.BITTER,
            texture=0.8,
        ),
        can_pick_up=True,
        is_mess=True,
        on_floor=True,
    )


def make_crumpled_paper(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="crumpled_paper",
        category=ObjectCategory.MESS,
        body_id=body_id,
        half_extents=(0.05, 0.05, 0.04),
        mass=0.02,
        color=(0.95, 0.95, 0.85, 1.0),
        sensory=SensoryProfile(texture=0.7),
        can_pick_up=True,
        is_mess=True,
        on_floor=True,
    )


def make_tv(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="tv",
        category=ObjectCategory.ENTERTAINMENT,
        body_id=body_id,
        half_extents=(0.6, 0.04, 0.35),
        mass=10.0,
        color=(0.05, 0.05, 0.05, 1.0),
        sensory=SensoryProfile(sound_level=60.0, sound_label="tv_audio"),
        can_pick_up=False,
        is_appliance=True,
        is_interactive=True,
        entertainment=0.6,
    )


def make_book(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="book",
        category=ObjectCategory.ENTERTAINMENT,
        body_id=body_id,
        half_extents=(0.1, 0.15, 0.02),
        mass=0.4,
        color=(0.3, 0.5, 0.8, 1.0),
        sensory=SensoryProfile(texture=0.2),
        can_pick_up=True,
        is_interactive=True,
        entertainment=0.7,
    )


def make_game_controller(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="game_controller",
        category=ObjectCategory.ENTERTAINMENT,
        body_id=body_id,
        half_extents=(0.08, 0.06, 0.025),
        mass=0.2,
        color=(0.15, 0.15, 0.15, 1.0),
        sensory=SensoryProfile(texture=0.4),
        can_pick_up=True,
        is_interactive=True,
        entertainment=0.8,
    )


def make_broom(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="broom",
        category=ObjectCategory.CLEANING,
        body_id=body_id,
        half_extents=(0.02, 0.02, 0.7),
        mass=0.8,
        color=(0.6, 0.4, 0.1, 1.0),
        sensory=SensoryProfile(smell_type=SmellType.CLEAN, smell_intensity=0.2),
        can_pick_up=True,
        is_interactive=True,
    )


def make_trash_bin(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="trash_bin",
        category=ObjectCategory.CLEANING,
        body_id=body_id,
        half_extents=(0.15, 0.15, 0.25),
        mass=1.5,
        color=(0.3, 0.3, 0.3, 1.0),
        sensory=SensoryProfile(
            smell_type=SmellType.GARBAGE,
            smell_intensity=0.5,
        ),
        can_pick_up=False,
        is_interactive=True,
        is_surface=True,
    )


# Furniture (static, heavy, not pickupable)
def make_bed(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="bed",
        category=ObjectCategory.FURNITURE,
        body_id=body_id,
        half_extents=(1.0, 1.0, 0.25),
        mass=0.0,
        color=(0.8, 0.7, 0.6, 1.0),
        can_pick_up=False,
        is_interactive=True,
        is_surface=True,
        energy_restore=0.8,
        sensory=SensoryProfile(texture=0.15, temperature=22.0),
    )


def make_sofa(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="sofa",
        category=ObjectCategory.FURNITURE,
        body_id=body_id,
        half_extents=(0.9, 0.45, 0.4),
        mass=0.0,
        color=(0.4, 0.3, 0.6, 1.0),
        can_pick_up=False,
        is_interactive=True,
        is_surface=True,
        sensory=SensoryProfile(texture=0.1),
    )


def make_dining_table(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="dining_table",
        category=ObjectCategory.FURNITURE,
        body_id=body_id,
        half_extents=(0.7, 0.45, 0.375),
        mass=0.0,
        color=(0.6, 0.4, 0.2, 1.0),
        can_pick_up=False,
        is_surface=True,
    )


def make_chair(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="chair",
        category=ObjectCategory.FURNITURE,
        body_id=body_id,
        half_extents=(0.22, 0.22, 0.44),
        mass=0.0,
        color=(0.5, 0.35, 0.15, 1.0),
        can_pick_up=False,
        is_interactive=True,
    )


def make_fridge(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="fridge",
        category=ObjectCategory.APPLIANCE,
        body_id=body_id,
        half_extents=(0.35, 0.35, 0.9),
        mass=0.0,
        color=(0.9, 0.9, 0.92, 1.0),
        can_pick_up=False,
        is_appliance=True,
        is_interactive=True,
        sensory=SensoryProfile(
            smell_type=SmellType.FOOD,
            smell_intensity=0.3,
            temperature=4.0,
        ),
    )


def make_stove(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="stove",
        category=ObjectCategory.APPLIANCE,
        body_id=body_id,
        half_extents=(0.35, 0.35, 0.45),
        mass=0.0,
        color=(0.2, 0.2, 0.2, 1.0),
        can_pick_up=False,
        is_appliance=True,
        is_interactive=True,
        sensory=SensoryProfile(temperature=200.0),
    )


def make_toilet(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="toilet",
        category=ObjectCategory.HYGIENE,
        body_id=body_id,
        half_extents=(0.2, 0.3, 0.4),
        mass=0.0,
        color=(0.95, 0.95, 0.95, 1.0),
        can_pick_up=False,
        is_interactive=True,
        sensory=SensoryProfile(
            smell_type=SmellType.CHEMICAL,
            smell_intensity=0.3,
            sound_label="toilet_flush",
        ),
    )


def make_shower(body_id: int = -1) -> WorldObject:
    return WorldObject(
        name="shower",
        category=ObjectCategory.HYGIENE,
        body_id=body_id,
        half_extents=(0.45, 0.45, 1.0),
        mass=0.0,
        color=(0.85, 0.9, 0.95, 0.8),
        can_pick_up=False,
        is_interactive=True,
        sensory=SensoryProfile(
            smell_type=SmellType.CLEAN,
            smell_intensity=0.4,
            sound_label="shower_water",
            temperature=38.0,
        ),
    )


# Registry of all archetypes for easy lookup
OBJECT_FACTORIES: Dict[str, callable] = {
    "apple":           make_apple,
    "pizza":           make_pizza,
    "water_bottle":    make_water_bottle,
    "dirty_sock":      make_dirty_sock,
    "crumpled_paper":  make_crumpled_paper,
    "tv":              make_tv,
    "book":            make_book,
    "game_controller": make_game_controller,
    "broom":           make_broom,
    "trash_bin":       make_trash_bin,
    "bed":             make_bed,
    "sofa":            make_sofa,
    "dining_table":    make_dining_table,
    "chair":           make_chair,
    "fridge":          make_fridge,
    "stove":           make_stove,
    "toilet":          make_toilet,
    "shower":          make_shower,
}


# ---------------------------------------------------------------------------
# Object Registry
# ---------------------------------------------------------------------------

class ObjectRegistry:
    """
    Tracks all WorldObject instances by their PyBullet body IDs.

    Provides spatial queries without requiring a physics world reference
    (positions are cached by the apartment at each step).
    """

    def __init__(self) -> None:
        self._objects: Dict[int, WorldObject] = {}
        self._positions: Dict[int, Tuple[float, float, float]] = {}

    def register(self, obj: WorldObject) -> None:
        if obj.body_id < 0:
            raise ValueError(f"Object '{obj.name}' has no valid body_id.")
        self._objects[obj.body_id] = obj

    def unregister(self, body_id: int) -> None:
        self._objects.pop(body_id, None)
        self._positions.pop(body_id, None)

    def update_position(
        self, body_id: int, pos: Tuple[float, float, float]
    ) -> None:
        self._positions[body_id] = pos

    def get(self, body_id: int) -> Optional[WorldObject]:
        return self._objects.get(body_id)

    def all(self) -> List[WorldObject]:
        return list(self._objects.values())

    def position_of(
        self, body_id: int
    ) -> Optional[Tuple[float, float, float]]:
        return self._positions.get(body_id)

    def distance(
        self,
        body_id: int,
        reference_pos: Tuple[float, float, float],
    ) -> float:
        pos = self._positions.get(body_id)
        if pos is None:
            return float("inf")
        dx = pos[0] - reference_pos[0]
        dy = pos[1] - reference_pos[1]
        dz = pos[2] - reference_pos[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def objects_within_radius(
        self,
        center: Tuple[float, float, float],
        radius: float,
        category: Optional[ObjectCategory] = None,
    ) -> List[WorldObject]:
        result = []
        for obj in self._objects.values():
            if category and obj.category != category:
                continue
            if self.distance(obj.body_id, center) <= radius:
                result.append(obj)
        return result

    def nearest(
        self,
        center: Tuple[float, float, float],
        category: Optional[ObjectCategory] = None,
        exclude_held: bool = True,
    ) -> Optional[WorldObject]:
        best_dist = float("inf")
        best_obj: Optional[WorldObject] = None
        for obj in self._objects.values():
            if exclude_held and obj.held_by_agent:
                continue
            if category and obj.category != category:
                continue
            d = self.distance(obj.body_id, center)
            if d < best_dist:
                best_dist = d
                best_obj = obj
        return best_obj

    def mess_count(self) -> int:
        """Number of mess objects currently on the floor."""
        return sum(
            1 for o in self._objects.values() if o.is_mess and o.on_floor
        )

    def clear(self) -> None:
        self._objects.clear()
        self._positions.clear()
