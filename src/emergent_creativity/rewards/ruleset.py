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
  is_sleeping, is_watching_tv, is_reading, is_playing_game

Supported operators in conditions:
  Comparisons: > < >= <= == !=
  Logic: and or not
  Numbers and booleans only — no imports, no function calls.
"""
from __future__ import annotations

import ast
import operator as _op
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default config path (relative to project root)
DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "rewards.yaml"

# Allowlist of AST node types permitted in condition expressions
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not,
    ast.Compare,
    ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.Constant,   # numbers, booleans
    ast.Name,       # variable references (validated against ctx keys)
    ast.Load,       # context node for Name lookups (read-only)
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
        return bool(eval(compile(tree, "<condition>", "eval"), {"__builtins__": {}}, ctx))  # noqa: S307
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rule parsing helpers
# ---------------------------------------------------------------------------

class Rule:
    """A single reward/punishment rule parsed from YAML."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: str        = data.get("name", "unnamed")
        self.event: str       = data.get("event", "")
        self.per_step: bool   = bool(data.get("per_step", False))
        self.condition: str   = data.get("condition", "")
        self.reward: float    = float(data.get("reward", 0.0))

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
        self._rules    = rules
        self.terminal  = terminal
        self.vitals_cfg = vitals_cfg

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
        ctx = {
            "hunger":         vitals.hunger,
            "energy":         vitals.energy,
            "bladder":        vitals.bladder,
            "happiness":      vitals.happiness,
            "mess_count":     float(registry.mess_count()),
            "idle_steps":     float(tenant.idle_steps),
            "is_sleeping":    tenant.is_sleeping,
            "is_watching_tv": tenant.is_watching_tv,
            "is_reading":     tenant.is_reading,
            "is_playing_game": tenant.is_playing_game,
        }

        total = 0.0
        info: dict = {}

        for rule in self._rules:
            earned = 0.0

            if rule.per_step:
                # Per-step rule: evaluate condition every step
                if rule.evaluate_condition(ctx):
                    earned = rule.reward
                    # For mess_count, scale by count
                    if "mess_count" in rule.condition:
                        earned *= max(1.0, ctx["mess_count"])
            else:
                # Event-triggered rule: fire once per matching event
                for event in tenant.events:
                    if rule.matches_event(event):
                        if rule.evaluate_condition(ctx):
                            earned += rule.reward

            if earned != 0.0:
                info[rule.name] = earned
                total += earned

        # Ongoing activity events
        if tenant.is_watching_tv:
            tenant.events.append("watching_tv")
        if tenant.is_reading:
            tenant.events.append("reading_book")
        if tenant.is_playing_game:
            tenant.events.append("playing_game")
        if tenant.is_sleeping:
            tenant.events.append("sleeping")

        return total, info

    def is_terminal(self, tenant: Any) -> bool:
        """Return True when the episode should end."""
        v = tenant.vitals
        term = self.terminal
        if v.hunger >= term.get("hunger_max", 1.0):
            return True
        if v.energy <= term.get("energy_min", 0.0):
            return True
        if tenant.total_steps >= term.get("max_steps", 50_000):
            return True
        return False

    @property
    def max_steps(self) -> int:
        return int(self.terminal.get("max_steps", 50_000))

    @property
    def rules(self) -> List[Rule]:
        return list(self._rules)
