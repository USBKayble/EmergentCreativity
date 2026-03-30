"""
apartment.py
============
Procedurally builds a fully-furnished apartment inside a :class:`PhysicsWorld`.

Layout
------
The apartment consists of four rooms arranged in a 2×2 grid:

    ┌──────────────┬──────────────┐
    │              │              │
    │  Living Room │   Bedroom    │
    │              │              │
    ├──────────────┼──────────────┤
    │              │              │
    │   Kitchen    │   Bathroom   │
    │              │              │
    └──────────────┴──────────────┘

Rooms are separated by thin walls with doorway openings.
All furniture is created as static boxes (mass=0).
Pickable objects (food, books, mess) are dynamic.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .physics_world import PhysicsWorld
from .objects import (
    ObjectRegistry,
    WorldObject,
    make_apple,
    make_pizza,
    make_water_bottle,
    make_dirty_sock,
    make_crumpled_paper,
    make_tv,
    make_book,
    make_game_controller,
    make_broom,
    make_trash_bin,
    make_bed,
    make_sofa,
    make_dining_table,
    make_chair,
    make_fridge,
    make_stove,
    make_toilet,
    make_shower,
)

# ---------------------------------------------------------------------------
# Room dimensions (metres)
# ---------------------------------------------------------------------------
ROOM_W = 5.0  # width  of each room
ROOM_D = 5.0  # depth  of each room
WALL_T = 0.15  # wall thickness
CEILING_H = 2.8  # room height
DOOR_W = 0.9  # doorway width
DOOR_H = 2.1  # doorway height

FLOOR_COLOR = (0.75, 0.70, 0.65, 1.0)
WALL_COLOR = (0.92, 0.90, 0.86, 1.0)
CEILING_COLOR = (0.97, 0.97, 0.97, 1.0)


# ---------------------------------------------------------------------------
# Apartment
# ---------------------------------------------------------------------------


class Apartment:
    """
    Builds and manages the entire 3-D apartment.

    Parameters
    ----------
    world : PhysicsWorld
        The physics engine instance (must already be started).
    registry : ObjectRegistry
        Shared object registry that the tenant & sensors will query.
    seed : int
        Random seed for initial mess / food placement.
    """

    def __init__(
        self,
        world: PhysicsWorld,
        registry: ObjectRegistry,
        seed: int = 42,
    ) -> None:
        self._world = world
        self._registry = registry
        self._rng = random.Random(seed)
        self._static_ids: List[int] = []  # walls, floors, ceilings
        self._furniture: List[WorldObject] = []
        self._items: List[WorldObject] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct the entire apartment from scratch."""
        self._build_floor()
        self._build_walls()
        self._build_furniture()
        self._place_items()

    def reset_items(self) -> None:
        """
        Respawn consumable / movable items to their starting positions.
        Does NOT rebuild walls or furniture.
        """
        for obj in list(self._items):
            if obj.body_id >= 0:
                self._world.remove_body(obj.body_id)
                self._registry.unregister(obj.body_id)
        self._items.clear()
        self._place_items()

    def sync_registry(self) -> None:
        """
        Update cached positions in the registry from the physics engine.
        Call this once per simulation step.
        """
        valid_items = []
        for obj in self._items:
            if obj.body_id < 0:
                continue
            if obj.consumed:
                continue
            try:
                pos, _ = self._world.get_position_orientation(obj.body_id)
                self._registry.update_position(obj.body_id, pos)
                if obj.can_pick_up and not obj.held_by_agent:
                    obj.on_floor = pos[2] < 0.15
                valid_items.append(obj)
            except Exception:
                pass
        self._items = valid_items

        for obj in self._furniture:
            if obj.body_id >= 0:
                try:
                    pos, _ = self._world.get_position_orientation(obj.body_id)
                    self._registry.update_position(obj.body_id, pos)
                    if obj.can_pick_up and not obj.held_by_agent:
                        obj.on_floor = pos[2] < 0.15
                except Exception:
                    pass

    @property
    def furniture(self) -> List[WorldObject]:
        return list(self._furniture)

    @property
    def items(self) -> List[WorldObject]:
        return list(self._items)

    # ------------------------------------------------------------------
    # Room origins (bottom-left corner of each room)
    # ------------------------------------------------------------------

    @staticmethod
    def living_room_origin() -> Tuple[float, float]:
        return (0.0, 0.0)

    @staticmethod
    def bedroom_origin() -> Tuple[float, float]:
        return (ROOM_W, 0.0)

    @staticmethod
    def kitchen_origin() -> Tuple[float, float]:
        return (0.0, ROOM_D)

    @staticmethod
    def bathroom_origin() -> Tuple[float, float]:
        return (ROOM_W, ROOM_D)

    @staticmethod
    def apartment_center() -> Tuple[float, float]:
        total_w = 2 * ROOM_W
        total_d = 2 * ROOM_D
        return (total_w / 2, total_d / 2)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _box(
        self,
        half_extents: Tuple[float, float, float],
        pos: Tuple[float, float, float],
        color: Tuple[float, float, float, float] = WALL_COLOR,
    ) -> int:
        bid = self._world.create_box(
            half_extents=half_extents,
            position=pos,
            mass=0.0,
            color=color,
            lateral_friction=0.8,
        )
        self._static_ids.append(bid)
        return bid

    def _build_floor(self) -> None:
        total_w = 2 * ROOM_W
        total_d = 2 * ROOM_D
        # Single large floor slab
        self._box(
            half_extents=(total_w / 2, total_d / 2, WALL_T / 2),
            pos=(total_w / 2, total_d / 2, -WALL_T / 2),
            color=FLOOR_COLOR,
        )
        # Ceiling
        self._box(
            half_extents=(total_w / 2, total_d / 2, WALL_T / 2),
            pos=(total_w / 2, total_d / 2, CEILING_H + WALL_T / 2),
            color=CEILING_COLOR,
        )

    def _build_walls(self) -> None:
        total_w = 2 * ROOM_W
        total_d = 2 * ROOM_D
        h2 = CEILING_H / 2
        hw = WALL_T / 2

        # Outer walls
        # South wall (y=0)
        self._box((total_w / 2 + WALL_T, hw, h2), (total_w / 2, 0.0, h2))
        # North wall (y=total_d)
        self._box((total_w / 2 + WALL_T, hw, h2), (total_w / 2, total_d, h2))
        # West wall (x=0)
        self._box((hw, total_d / 2, h2), (0.0, total_d / 2, h2))
        # East wall (x=total_w)
        self._box((hw, total_d / 2, h2), (total_w, total_d / 2, h2))

        # Interior wall between Living Room and Bedroom (x=ROOM_W, y=0..ROOM_D)
        # With a doorway opening
        self._wall_with_door(
            axis="x",
            fixed_coord=ROOM_W,
            span_start=0.0,
            span_end=ROOM_D,
            door_center=ROOM_D / 2,
        )

        # Interior wall between Kitchen and Bathroom (x=ROOM_W, y=ROOM_D..2*ROOM_D)
        self._wall_with_door(
            axis="x",
            fixed_coord=ROOM_W,
            span_start=ROOM_D,
            span_end=2 * ROOM_D,
            door_center=ROOM_D + ROOM_D / 2,
        )

        # Interior wall between Living Room and Kitchen (y=ROOM_D, x=0..ROOM_W)
        self._wall_with_door(
            axis="y",
            fixed_coord=ROOM_D,
            span_start=0.0,
            span_end=ROOM_W,
            door_center=ROOM_W / 2,
        )

        # Interior wall between Bedroom and Bathroom (y=ROOM_D, x=ROOM_W..2*ROOM_W)
        self._wall_with_door(
            axis="y",
            fixed_coord=ROOM_D,
            span_start=ROOM_W,
            span_end=2 * ROOM_W,
            door_center=ROOM_W + ROOM_W / 2,
        )

    def _wall_with_door(
        self,
        axis: str,
        fixed_coord: float,
        span_start: float,
        span_end: float,
        door_center: float,
    ) -> None:
        """
        Build a wall segment along *axis* at *fixed_coord*,
        leaving a doorway gap centred at *door_center*.
        """
        hw = WALL_T / 2
        dw2 = DOOR_W / 2
        h2 = CEILING_H / 2

        door_left = door_center - dw2
        door_right = door_center + dw2

        # Above doorway lintel
        lintel_h = (CEILING_H - DOOR_H) / 2

        for seg_start, seg_end in [
            (span_start, door_left),
            (door_right, span_end),
        ]:
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            seg_mid = (seg_start + seg_end) / 2
            if axis == "x":
                self._box(
                    (hw, seg_len / 2, h2),
                    (fixed_coord, seg_mid, h2),
                )
            else:
                self._box(
                    (seg_len / 2, hw, h2),
                    (seg_mid, fixed_coord, h2),
                )

        # Lintel above door
        if lintel_h > 0:
            if axis == "x":
                self._box(
                    (hw, dw2, lintel_h / 2),
                    (fixed_coord, door_center, DOOR_H + lintel_h / 2),
                )
            else:
                self._box(
                    (dw2, hw, lintel_h / 2),
                    (door_center, fixed_coord, DOOR_H + lintel_h / 2),
                )

    # ------------------------------------------------------------------
    # Furniture placement
    # ------------------------------------------------------------------

    def _build_furniture(self) -> None:
        self._build_living_room_furniture()
        self._build_bedroom_furniture()
        self._build_kitchen_furniture()
        self._build_bathroom_furniture()

    def _spawn_furniture(
        self,
        obj: WorldObject,
        position: Tuple[float, float, float],
        yaw: float = 0.0,
    ) -> None:
        orn = self._world.euler_to_quaternion(0.0, 0.0, yaw)
        # Place bottom of object at z level (half_extents[2] up)
        pos = (position[0], position[1], position[2] + obj.half_extents[2])
        obj.body_id = self._world.create_box(
            half_extents=obj.half_extents,
            position=pos,
            orientation=orn,
            mass=obj.mass,
            color=obj.color,
        )
        self._registry.register(obj)
        self._registry.update_position(obj.body_id, pos)
        self._furniture.append(obj)

    def _build_living_room_furniture(self) -> None:
        ox, oy = self.living_room_origin()
        cx = ox + ROOM_W / 2

        # Sofa facing TV
        sofa = make_sofa()
        self._spawn_furniture(sofa, (cx, oy + 1.2, 0.0), yaw=0.0)

        # TV on north wall
        tv = make_tv()
        self._spawn_furniture(tv, (cx, oy + ROOM_D - 0.5, 0.0))

        # Coffee table in front of sofa
        table = make_dining_table()
        table.name = "coffee_table"
        table.half_extents = (0.5, 0.3, 0.2)
        self._spawn_furniture(table, (cx, oy + 2.2, 0.0))

    def _build_bedroom_furniture(self) -> None:
        ox, oy = self.bedroom_origin()
        cx, cy = ox + ROOM_W / 2, oy + ROOM_D / 2

        # Bed against south wall
        bed = make_bed()
        self._spawn_furniture(bed, (cx, oy + 1.2, 0.0))

        # Desk on east wall
        desk = make_dining_table()
        desk.name = "desk"
        desk.half_extents = (0.6, 0.3, 0.375)
        self._spawn_furniture(desk, (ox + ROOM_W - 0.8, cy, 0.0))

        desk_chair = make_chair()
        self._spawn_furniture(desk_chair, (ox + ROOM_W - 0.8, cy - 0.7, 0.0))

    def _build_kitchen_furniture(self) -> None:
        ox, oy = self.kitchen_origin()
        cx, cy = ox + ROOM_W / 2, oy + ROOM_D / 2

        # Fridge against east wall
        fridge = make_fridge()
        self._spawn_furniture(fridge, (ox + ROOM_W - 0.5, oy + ROOM_D - 1.0, 0.0))

        # Stove next to fridge
        stove = make_stove()
        self._spawn_furniture(stove, (ox + ROOM_W - 0.5, oy + ROOM_D - 2.0, 0.0))

        # Dining table in centre
        d_table = make_dining_table()
        self._spawn_furniture(d_table, (cx, cy, 0.0))

        # Two chairs
        for dx in [-0.9, 0.9]:
            ch = make_chair()
            self._spawn_furniture(ch, (cx + dx, cy, 0.0))

        # Trash bin
        trash = make_trash_bin()
        self._spawn_furniture(trash, (ox + 0.3, oy + 0.3, 0.0))

        # Broom in corner
        broom = make_broom()
        broom.half_extents = (0.02, 0.02, 0.7)
        self._spawn_furniture(broom, (ox + 0.2, oy + ROOM_D - 0.3, 0.0))

    def _build_bathroom_furniture(self) -> None:
        ox, oy = self.bathroom_origin()

        # Toilet
        toilet = make_toilet()
        self._spawn_furniture(toilet, (ox + 0.4, oy + 0.4, 0.0))

        # Shower stall
        shower = make_shower()
        self._spawn_furniture(shower, (ox + ROOM_W - 0.6, oy + 0.7, 0.0))

    # ------------------------------------------------------------------
    # Item placement (food, mess, books)
    # ------------------------------------------------------------------

    def _place_items(self) -> None:
        self._place_food()
        self._place_mess()
        self._place_entertainment_items()

    def _spawn_item(
        self,
        obj: WorldObject,
        position: Tuple[float, float, float],
    ) -> None:
        pos = (position[0], position[1], position[2] + obj.half_extents[2] + 0.01)
        obj.body_id = self._world.create_box(
            half_extents=obj.half_extents,
            position=pos,
            mass=max(obj.mass, 0.01),  # ensure dynamic
            color=obj.color,
            lateral_friction=0.6,
            restitution=0.1,
        )
        self._registry.register(obj)
        self._registry.update_position(obj.body_id, pos)
        self._items.append(obj)

    def _place_food(self) -> None:
        # Apple on kitchen dining table
        kx, ky = self.kitchen_origin()
        table_top_z = 0.375 * 2 + 0.01  # table half_extents z * 2
        apple = make_apple()
        self._spawn_item(apple, (kx + ROOM_W / 2, ky + ROOM_D / 2, table_top_z))

        # Pizza on kitchen dining table
        pizza = make_pizza()
        self._spawn_item(pizza, (kx + ROOM_W / 2 + 0.3, ky + ROOM_D / 2, table_top_z))

        # Water bottle on coffee table (living room)
        lx, ly = self.living_room_origin()
        coffee_top_z = 0.2 * 2 + 0.01
        water = make_water_bottle()
        self._spawn_item(water, (lx + ROOM_W / 2, ly + 2.2, coffee_top_z))

    def _place_mess(self) -> None:
        # A few mess items scattered in living room
        lx, ly = self.living_room_origin()
        mess_positions = [
            (lx + 1.5, ly + 1.5),
            (lx + 3.0, ly + 2.0),
            (lx + 2.5, ly + 1.0),
        ]
        for i, (mx, my) in enumerate(mess_positions):
            if i % 2 == 0:
                mess = make_dirty_sock()
            else:
                mess = make_crumpled_paper()
            mess.name = f"{mess.name}_{i}"
            self._spawn_item(mess, (mx, my, 0.0))
            mess.on_floor = True

    def _place_entertainment_items(self) -> None:
        # Book on desk in bedroom
        bx, by = self.bedroom_origin()
        desk_top_z = 0.375 * 2 + 0.01
        book = make_book()
        self._spawn_item(book, (bx + ROOM_W - 0.8, by + ROOM_D / 2, desk_top_z))

        # Game controller on coffee table (living room)
        lx, ly = self.living_room_origin()
        controller = make_game_controller()
        self._spawn_item(controller, (lx + ROOM_W / 2 - 0.2, ly + 2.2, 0.2 * 2 + 0.01))
