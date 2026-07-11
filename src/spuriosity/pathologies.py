"""
Pathology classes — pluggable, composable modifications to a PanelGenerator's
data-generating process.

v1 ships three pathologies: StructuralBreak, Confounder, SelectionBias.
Unit root is deferred to v1.1.

Each Pathology subclass must:
  1. Implement the DGP modification logic (applied during PanelGenerator.generate()).
  2. Report what GroundTruth fields it populates/affects.
  3. Declare known conflicts with other pathology types for validate_combo()
     (as warnings, not hard errors — v1 policy is permissive).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Pathology(ABC):
    """Base class for all injectable pathologies."""

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError

    def conflicts_with(self, other: "Pathology") -> str | None:
        """Return a warning message if this pathology likely conflicts with
        `other` when composed, or None if no known conflict."""
        return None


class StructuralBreak(Pathology):
    """Injects a regime change (mean shift, variance shift, or relationship
    shift) at a specified period."""

    def __init__(self, period: int, target: str, kind: str, magnitude: float) -> None:
        self.period = period
        self.target = target
        self.kind = kind
        self.magnitude = magnitude

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError


class Confounder(Pathology):
    """Injects an unobserved (or observed) variable that confounds a feature
    and an outcome, inducing a spurious relationship."""

    def __init__(self, feature: str, outcome: str, strength: float, observed: bool = False) -> None:
        self.feature = feature
        self.outcome = outcome
        self.strength = strength
        self.observed = observed

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError


class SelectionBias(Pathology):
    """Applies non-random sample selection according to a boolean rule,
    evaluated via a constrained pandas.eval (see CONTRIBUTING.md)."""

    def __init__(self, rule: str, drop_prob: float) -> None:
        self.rule = rule
        self.drop_prob = drop_prob

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError


def validate_combo(pathologies: list[Pathology]) -> list[str]:
    """Check a list of pathologies for likely conflicts. Returns a list of
    warning messages (empty if none). Never raises — v1 policy is permissive."""
    raise NotImplementedError
