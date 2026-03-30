"""
ruleset.py
==========
Loads reward / punishment rules from ``config/rewards.yaml`` and evaluates
them each simulation step.

The YAML format is intentionally simple so non-programmers can tweak it:

    rules:
      - name: eat_food
        event: ate_food
        reward: 10.0
      - name: hungry_penalty
        per_step: true
        condition: "hunger > 0.6"
        reward: -0.5

Supported condition variables (all float/int/bool):
  hunger, energy, bladder, happiness,
  mess_count, idle_steps,
  is_sleeping, is_watching_tv, is_reading, is_playing_game,
  in_kitchen, in_bathroom, in_bedroom, in_living_room,
  smell_food, smell_garbage, smell_floral,
  near_food, near_bed, near_toilet, near_tv, near_book, near_game

Supported operators in conditions:
  Comparisons: > < >= <= == !=
  Logic: and or not
  Numbers and booleans only — no imports, no function calls.

Homeostatic Reward System (Option C):
====================================
Based on Keramati & Gutkin (2014) "Homeostatic reinforcement learning for
integrating reward collection and physiological stability" - eLife.

Key principles:
1. Reward NEED REDUCTION, not penalty for having needs
2. S-shaped reward curves prevent gaming (diminishing marginal returns)
3. Homeostatic zone rewards encourage maintaining optimal vital ranges
4. Drive function D(H) = distance from optimal setpoint

The mathematical proof shows: reward maximization ≈ homeostatic stability
when reward = D(H_t) - D(H_{t+1}) (drive reduction)
"""

from __future__ import annotations

import ast
import operator as _op
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

import numpy as np


# Room boundaries (matching apartment.py)
ROOM_W = 5.0
ROOM_D = 5.0


# Default config path (relative to project root)
DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "rewards.yaml"

# Allowlist of AST node types permitted in condition expressions
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Gt,
    ast.Lt,
    ast.GtE,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Constant,  # numbers, booleans
    ast.Name,  # variable references (validated against ctx keys)
    ast.Load,  # context node for Name lookups (read-only)
)


def _safe_eval_condition(expr: str, ctx: Dict[str, Any]) -> bool:
    """
    Parse and evaluate a simple boolean condition safely.

    Only allows comparisons, boolean logic, numeric literals, and references
    to variables present in *ctx*.  No function calls, imports, or attribute
    access are permitted.

    Returns False if parsing or evaluation fails.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    # Walk the AST and reject any disallowed node types
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Name) and node.id not in ctx:
            return False

    # Safe to evaluate with a restricted namespace
    try:
        return bool(
            eval(compile(tree, "<condition>", "eval"), {"__builtins__": {}}, ctx)
        )  # noqa: S307
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rule parsing helpers
# ---------------------------------------------------------------------------


class Rule:
    """A single reward/punishment rule parsed from YAML."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: str = data.get("name", "unnamed")
        self.event: str = data.get("event", "")
        self.per_step: bool = bool(data.get("per_step", False))
        self.condition: str = data.get("condition", "")
        self.reward: float = float(data.get("reward", 0.0))

    def matches_event(self, event: str) -> bool:
        return bool(self.event) and self.event == event

    def evaluate_condition(self, ctx: Dict[str, Any]) -> bool:
        """
        Evaluate the rule's condition expression using *ctx* as the variable
        namespace.  Returns True when no condition is specified.
        """
        if not self.condition:
            return True
        return _safe_eval_condition(self.condition, ctx)


# ---------------------------------------------------------------------------
# Reward evaluator
# ---------------------------------------------------------------------------


class RewardEvaluator:
    """
    Evaluates all rules against the current simulation state each step.

    Usage
    -----
    ::

        evaluator = RewardEvaluator.from_yaml()
        ...
        # each sim step:
        reward, info = evaluator.evaluate(tenant, registry)
    """

    def __init__(self, rules: List[Rule], terminal: dict, vitals_cfg: dict) -> None:
        self._rules = rules
        self.terminal = terminal
        self.vitals_cfg = vitals_cfg
        self._prev_pos: Optional[Tuple[float, float, float]] = None
        self._prev_smell: Optional[Dict[str, float]] = None
        self._prev_proximity: Dict[str, float] = {
            "near_food": 0.0,
            "near_bed": 0.0,
            "near_toilet": 0.0,
            "near_tv": 0.0,
            "near_book": 0.0,
            "near_game": 0.0,
        }
        self._prev_vitals: Optional[Dict[str, float]] = None
        self._reward_history: List[float] = []
        self._running_mean: float = 0.0
        self._running_var: float = 1.0
        self._reward_count: int = 0
        self._gamma: float = 0.99
        self._epsilon: float = 1e-8
        self._action_counts: Dict[str, int] = {}
        self._action_decay: float = 0.85

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "RewardEvaluator":
        """Load from a YAML file (defaults to ``config/rewards.yaml``)."""
        config_path = Path(path) if path else DEFAULT_CONFIG
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        rules = [Rule(r) for r in data.get("rules", [])]
        terminal = data.get("terminal", {})
        vitals_cfg = data.get("vitals", {})
        return cls(rules, terminal, vitals_cfg)

    @classmethod
    def from_dict(cls, data: dict) -> "RewardEvaluator":
        """Build from an already-loaded dict (useful for testing)."""
        rules = [Rule(r) for r in data.get("rules", [])]
        terminal = data.get("terminal", {})
        vitals_cfg = data.get("vitals", {})
        return cls(rules, terminal, vitals_cfg)

    # ------------------------------------------------------------------
    # Room detection helpers
    # ------------------------------------------------------------------

    def _get_room(self, pos: Tuple[float, float, float]) -> str:
        x, y = pos[0], pos[1]
        if x < ROOM_W and y < ROOM_D:
            return "living_room"
        elif x >= ROOM_W and y < ROOM_D:
            return "bedroom"
        elif x < ROOM_W and y >= ROOM_D:
            return "kitchen"
        else:
            return "bathroom"

    def _is_in_room(self, pos: Tuple[float, float, float], room: str) -> bool:
        return self._get_room(pos) == room

    def _get_proximity_to_objects(
        self, pos: Tuple[float, float, float], registry: Any
    ) -> Dict[str, float]:
        result = {
            "near_food": 0.0,
            "near_bed": 0.0,
            "near_toilet": 0.0,
            "near_tv": 0.0,
            "near_book": 0.0,
            "near_game": 0.0,
        }
        if registry is None:
            return result

        for obj in registry.all():
            dist = registry.distance(obj.body_id, pos)
            if dist > 3.0:
                continue
            proximity = max(0.0, 1.0 - dist / 3.0)

            if obj.name in ("fridge", "stove", "apple", "pizza", "water_bottle"):
                result["near_food"] = max(result["near_food"], proximity)
            elif obj.name == "bed":
                result["near_bed"] = max(result["near_bed"], proximity)
            elif obj.name == "toilet":
                result["near_toilet"] = max(result["near_toilet"], proximity)
            elif obj.name == "tv":
                result["near_tv"] = max(result["near_tv"], proximity)
            elif "book" in obj.name:
                result["near_book"] = max(result["near_book"], proximity)
            elif "game" in obj.name:
                result["near_game"] = max(result["near_game"], proximity)

        return result

    def _get_smell_intensities(
        self, smell_array: Optional[np.ndarray]
    ) -> Dict[str, float]:
        if smell_array is None or len(smell_array) < 7:
            return {
                "smell_food": 0.0,
                "smell_garbage": 0.0,
                "smell_floral": 0.0,
            }
        return {
            "smell_food": float(smell_array[1]),
            "smell_garbage": float(smell_array[2]),
            "smell_floral": float(smell_array[5]),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, tenant: Any, registry: Any) -> tuple[float, dict]:
        """
        Calculate the total reward for the current step.

        Parameters
        ----------
        tenant   : Tenant     – the agent with vitals, events, flags
        registry : ObjectRegistry

        Returns
        -------
        total_reward : float
        info         : dict  – per-rule breakdown
        """
        vitals = tenant.vitals
        pos = tenant.get_position()

        room = self._get_room(pos)
        proximity = self._get_proximity_to_objects(pos, registry)

        smell_intensities = self._get_smell_intensities(
            tenant._sensors.smell.observe(pos) if hasattr(tenant, "_sensors") else None
        )

        ctx = {
            "hunger": vitals.hunger,
            "energy": vitals.energy,
            "bladder": vitals.bladder,
            "happiness": vitals.happiness,
            "mess_count": float(registry.mess_count()),
            "idle_steps": float(tenant.idle_steps),
            "is_sleeping": tenant.is_sleeping,
            "is_watching_tv": tenant.is_watching_tv,
            "is_reading": tenant.is_reading,
            "is_playing_game": tenant.is_playing_game,
            "in_kitchen": room == "kitchen",
            "in_bathroom": room == "bathroom",
            "in_bedroom": room == "bedroom",
            "in_living_room": room == "living_room",
            "near_food": proximity["near_food"],
            "near_bed": proximity["near_bed"],
            "near_toilet": proximity["near_toilet"],
            "near_tv": proximity["near_tv"],
            "near_book": proximity["near_book"],
            "near_game": proximity["near_game"],
            "smell_food": smell_intensities["smell_food"],
            "smell_garbage": smell_intensities["smell_garbage"],
            "smell_floral": smell_intensities["smell_floral"],
        }

        total = 0.0
        info: dict = {}

        critical_mult = self._get_critical_need_multiplier(vitals)
        action_type = "idle"
        if tenant.is_sleeping:
            action_type = "sleep"
        elif tenant.is_watching_tv:
            action_type = "entertainment"
        elif any(e in ["ate_food"] for e in tenant.events):
            action_type = "eating"
        elif any(e in ["used_toilet"] for e in tenant.events):
            action_type = "bathroom"

        decay = self._get_action_decay(action_type)

        for rule in self._rules:
            earned = 0.0

            if rule.per_step:
                has_event = bool(rule.event)
                if has_event:
                    if any(rule.matches_event(e) for e in tenant.events):
                        if rule.evaluate_condition(ctx):
                            earned = rule.reward
                else:
                    if rule.evaluate_condition(ctx):
                        earned = rule.reward
                        if "mess_count" in rule.condition:
                            earned *= max(1.0, ctx["mess_count"])
            else:
                for event in tenant.events:
                    if rule.matches_event(event):
                        if rule.evaluate_condition(ctx):
                            earned += rule.reward

            if earned != 0.0:
                earned *= decay
                info[rule.name] = earned
                total += earned

        self._update_action_count(action_type)

        guidance_reward, guidance_info = self._compute_guidance_rewards(
            vitals, smell_intensities, room, proximity, pos, critical_mult, decay
        )
        total += guidance_reward
        info.update(guidance_info)

        self._prev_pos = pos
        self._prev_smell = smell_intensities
        self._prev_proximity = proximity.copy()

        if tenant.is_watching_tv:
            tenant.events.append("watching_tv")
        if tenant.is_reading:
            tenant.events.append("reading_book")
        if tenant.is_playing_game:
            tenant.events.append("playing_game")
        if tenant.is_sleeping:
            tenant.events.append("sleeping")

        return total, info

    def _compute_drive(
        self, vital_value: float, optimal: float, exponent: float = 2.0
    ) -> float:
        distance = abs(vital_value - optimal) ** exponent
        return distance

    def _potential(self, vitals: Any) -> float:
        current = {
            "hunger": vitals.hunger,
            "energy": vitals.energy,
            "bladder": vitals.bladder,
            "happiness": vitals.happiness,
        }
        return self._potential_from_dict(current)

    def _potential_from_dict(self, current: Dict[str, float]) -> float:
        optimal = {
            "hunger": 0.3,
            "energy": 0.8,
            "bladder": 0.2,
            "happiness": 0.7,
        }
        total_drive = 0.0
        for name in current:
            dist = abs(current[name] - optimal[name])
            total_drive += dist**2
        return 1.0 / (1.0 + total_drive)

    def _normalize_reward(self, reward: float) -> float:
        if not np.isfinite(reward):
            return 0.0
        self._reward_history.append(reward)
        if len(self._reward_history) > 10000:
            self._reward_history = self._reward_history[-10000:]
        self._reward_count += 1
        if self._reward_count < 10:
            return reward
        old_mean = self._running_mean
        delta = reward - old_mean
        self._running_mean += self._gamma * delta
        self._running_var = self._gamma * self._running_var + (1 - self._gamma) * (
            delta * (reward - self._running_mean)
        )
        std = np.sqrt(max(self._running_var, 0.01) + self._epsilon)
        normalized = (reward - self._running_mean) / std
        return np.clip(normalized, -10.0, 10.0)

    def _sigmoid_reward(
        self,
        current_value: float,
        delta: float,
        optimal: float,
        steepness: float = 8.0,
        max_reward: float = 0.5,
    ) -> float:
        new_value = current_value + delta
        if new_value > optimal:
            normalized = (new_value - optimal) / (1.0 - optimal)
            sig = 1.0 / (1.0 + np.exp(-steepness * (normalized - 0.5)))
            base_reward = max_reward * (1.0 - sig)
        else:
            normalized = (optimal - new_value) / optimal
            sig = 1.0 / (1.0 + np.exp(-steepness * (normalized - 0.5)))
            base_reward = max_reward * sig
        if new_value > 1.0 or new_value < 0.0:
            base_reward -= 0.2
        return base_reward

    def _get_action_decay(self, action_type: str) -> float:
        count = self._action_counts.get(action_type, 0)
        return self._action_decay ** min(count, 20)

    def _update_action_count(self, action_type: str) -> None:
        self._action_counts[action_type] = self._action_counts.get(action_type, 0) + 1

    def _get_critical_need_multiplier(self, vitals: Any) -> Dict[str, float]:
        current = {
            "hunger": vitals.hunger,
            "energy": vitals.energy,
            "bladder": vitals.bladder,
            "happiness": vitals.happiness,
        }
        critical_multiplier = {}
        max_criticality = 0.0
        critical_need = None
        optimal = {"hunger": 0.3, "energy": 0.8, "bladder": 0.2, "happiness": 0.7}
        for name, val in current.items():
            criticality = 0.0
            opt = optimal.get(name, 0.5)
            distance = abs(val - opt)
            criticality = distance / 0.5
            if criticality > max_criticality:
                max_criticality = criticality
                critical_need = name
        if max_criticality > 0.3 and critical_need:
            for name in current.keys():
                if name == critical_need:
                    critical_multiplier[name] = 1.0
                else:
                    critical_multiplier[name] = 0.3
        else:
            for name in current.keys():
                critical_multiplier[name] = 1.0
        return critical_multiplier

    def _compute_homeostatic_reward(
        self,
        vitals: Any,
        critical_mult: Optional[Dict[str, float]] = None,
        action_decay: float = 1.0,
    ) -> tuple[float, dict]:
        reward = 0.0
        info = {}
        critical_mult = critical_mult or {
            k: 1.0 for k in ["hunger", "energy", "bladder", "happiness"]
        }
        action_decay = action_decay or 1.0
        current_vitals = {
            "hunger": vitals.hunger,
            "energy": vitals.energy,
            "bladder": vitals.bladder,
            "happiness": vitals.happiness,
        }
        optimal = {
            "hunger": 0.3,
            "energy": 0.8,
            "bladder": 0.2,
            "happiness": 0.7,
        }
        weights = {
            "hunger": 1.0,
            "energy": 1.0,
            "bladder": 0.8,
            "happiness": 0.6,
        }
        if self._prev_vitals is not None:
            for vital_name in current_vitals:
                prev_val = self._prev_vitals.get(vital_name)
                if prev_val is None:
                    prev_val = current_vitals[vital_name]
                curr_val = current_vitals[vital_name]
                delta = prev_val - curr_val
                if delta > 0:
                    rw = self._sigmoid_reward(
                        float(prev_val),
                        -float(delta),
                        optimal[vital_name],
                        max_reward=0.4,
                    )
                    rw *= weights[vital_name] * critical_mult.get(vital_name, 1.0)
                    rw *= action_decay
                    reward += rw
                    info[f"need_reduction_{vital_name}"] = rw
        zone_rewards = {}
        for vital_name in current_vitals:
            val = current_vitals[vital_name]
            opt = optimal[vital_name]
            if opt - 0.15 < val < opt + 0.15:
                zone_rewards[vital_name] = (
                    0.15
                    * weights[vital_name]
                    * critical_mult.get(vital_name, 1.0)
                    * action_decay
                )
                reward += zone_rewards[vital_name]
        if zone_rewards:
            info["homeostatic_zone"] = sum(zone_rewards.values())

        critical_penalty = 0.0
        for vital_name, val in current_vitals.items():
            is_critical = False
            if vital_name in ("hunger", "bladder"):
                if val >= 0.9:
                    is_critical = True
            elif vital_name == "energy":
                if val <= 0.1:
                    is_critical = True
            elif vital_name == "happiness":
                if val <= 0.1:
                    is_critical = True

            if is_critical:
                severity = abs(val - 0.5) * 2
                non_linear = severity**3
                critical_penalty -= 0.2 * non_linear * weights.get(vital_name, 1.0)
        if critical_penalty != 0.0:
            reward += critical_penalty
            info["critical_neglect_penalty"] = critical_penalty

        self._prev_vitals = current_vitals
        return reward, info

    def _compute_guidance_rewards(
        self,
        vitals: Any,
        smell_intensities: Dict[str, float],
        room: str,
        proximity: Dict[str, float],
        pos: Optional[Tuple[float, float, float]] = None,
        critical_mult: Optional[Dict[str, float]] = None,
        action_decay: float = 1.0,
    ) -> tuple[float, dict]:
        reward = 0.0
        info = {}
        critical_mult = critical_mult or {
            k: 1.0 for k in ["hunger", "energy", "bladder", "happiness"]
        }
        action_decay = action_decay or 1.0
        homeo_reward, homeo_info = self._compute_homeostatic_reward(
            vitals, critical_mult, action_decay
        )
        reward += homeo_reward
        info.update(homeo_info)

        if pos is None or self._prev_pos is None:
            return reward, info

        current = {
            "hunger": vitals.hunger,
            "energy": vitals.energy,
            "bladder": vitals.bladder,
            "happiness": vitals.happiness,
        }
        optimal = {
            "hunger": 0.3,
            "energy": 0.8,
            "bladder": 0.2,
            "happiness": 0.7,
        }

        goals = [
            ("food", "near_food", current["hunger"], optimal["hunger"], 0.6, 1.0),
            (
                "bed",
                "near_bed",
                1.0 - current["energy"],
                1.0 - optimal["energy"],
                0.4,
                0.6,
            ),
            ("toilet", "near_toilet", current["bladder"], optimal["bladder"], 0.6, 0.9),
            (
                "entertainment",
                "near_tv",
                1.0 - current["happiness"],
                1.0 - optimal["happiness"],
                0.3,
                0.4,
            ),
        ]

        for goal_name, prox_key, need, optimal_need, threshold, base_reward in goals:
            if need < threshold:
                continue

            urgency = (need - optimal_need) / (1.0 - optimal_need + 0.01)
            urgency = min(1.0, max(0.0, urgency))

            prev_prox = self._prev_proximity.get(prox_key, 0.0)
            curr_prox = proximity.get(prox_key, 0.0)
            prox_delta = curr_prox - prev_prox

            if prox_delta > 0:
                direction_factor = 1.0
            elif prox_delta < 0:
                direction_factor = -0.5
            else:
                direction_factor = 0.0

            distance_factor = 1.0 - curr_prox

            rw = (
                direction_factor
                * distance_factor
                * urgency
                * base_reward
                * action_decay
            )
            reward += rw
            info[f"goal_{goal_name}"] = rw

        return reward, info

    def is_terminal(self, tenant: Any) -> bool:
        """Return True when the episode should end."""
        return False

    @property
    def max_steps(self) -> int:
        return int(self.terminal.get("max_steps", 50_000))

    @property
    def rules(self) -> List[Rule]:
        return list(self._rules)
