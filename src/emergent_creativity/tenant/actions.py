"""
actions.py
==========
Defines the discrete action space available to the tenant agent.

Action space overview
---------------------
The tenant has 13 discrete actions (0–12):

  0  IDLE            – do nothing
  1  MOVE_FORWARD    – step forward ~0.1 m
  2  MOVE_BACKWARD   – step backward ~0.1 m
  3  MOVE_LEFT       – strafe left ~0.1 m
  4  MOVE_RIGHT      – strafe right ~0.1 m
  5  TURN_LEFT       – rotate left 10°
  6  TURN_RIGHT      – rotate right 10°
  7  PICK_UP         – pick up nearest reachable object
  8  PUT_DOWN        – place held object on nearest surface / floor
  9  INTERACT        – interact with nearest interactive object (toggle TV, open fridge …)
  10 EAT             – eat held food item
  11 SLEEP           – lie down and sleep (must be on / near bed / sofa)
  12 USE_BATHROOM    – use toilet (must be near toilet)

All movement is applied via velocity impulse so that the physics engine
handles collisions naturally.
"""

from enum import IntEnum


class Action(IntEnum):
    IDLE          = 0
    MOVE_FORWARD  = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT     = 3
    MOVE_RIGHT    = 4
    TURN_LEFT     = 5
    TURN_RIGHT    = 6
    PICK_UP       = 7
    PUT_DOWN      = 8
    INTERACT      = 9
    EAT           = 10
    SLEEP         = 11
    USE_BATHROOM  = 12


N_ACTIONS = len(Action)

# Human-readable labels for UI overlay
ACTION_LABELS: dict[int, str] = {
    Action.IDLE:          "Idle",
    Action.MOVE_FORWARD:  "Move Forward",
    Action.MOVE_BACKWARD: "Move Backward",
    Action.MOVE_LEFT:     "Strafe Left",
    Action.MOVE_RIGHT:    "Strafe Right",
    Action.TURN_LEFT:     "Turn Left",
    Action.TURN_RIGHT:    "Turn Right",
    Action.PICK_UP:       "Pick Up",
    Action.PUT_DOWN:      "Put Down",
    Action.INTERACT:      "Interact",
    Action.EAT:           "Eat",
    Action.SLEEP:         "Sleep",
    Action.USE_BATHROOM:  "Use Bathroom",
}

# Movement parameters
MOVE_SPEED      = 1.5    # m/s linear speed
TURN_SPEED_DEG  = 10.0   # degrees per action step
REACH_DISTANCE  = 1.5    # metres – how close the agent must be to interact
SLEEP_DISTANCE  = 1.0    # metres – how close to a bed/sofa to sleep
TOILET_DISTANCE = 1.0    # metres – how close to the toilet to use it
