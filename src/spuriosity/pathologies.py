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
from typing import Literal

import numpy as np
import pandas as pd

from spuriosity.ground_truth import BreakInfo

_STRUCTURAL_BREAK_KINDS = ("mean_shift", "variance_shift", "coefficient_shift")


class Pathology(ABC):
    """Base class for all injectable pathologies."""

    @abstractmethod
    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError

    def conflicts_with(self, other: "Pathology") -> str | None:
        """Return a warning message if this pathology likely conflicts with
        `other` when composed, or None if no known conflict."""
        return None


class StructuralBreak(Pathology):
    """Injects a regime change at a specified period.

    Three kinds are supported:

    - ``"mean_shift"``: adds ``magnitude`` to the outcome's mean from
      ``period`` onward.
    - ``"variance_shift"``: multiplies the outcome noise standard deviation
      by ``magnitude`` from ``period`` onward (``magnitude`` is a
      multiplicative factor, not an additive one -- e.g. ``2.0`` doubles
      the noise std post-break).
    - ``"coefficient_shift"``: changes the true coefficient on
      ``coefficient_target`` (a design-matrix column name) to
      ``magnitude`` from ``period`` onward, replacing whatever value it had
      pre-break. Only valid when the outcome was specified via ``formula=``
      (coefficient_shift has no meaning for an ``fn=``-specified DGP).

    ``target`` names the outcome column being broken (almost always the
    generator's outcome name, e.g. ``"y"``); it is recorded in
    ``BreakInfo`` for ground-truth bookkeeping but is not independently
    validated against the generator's outcome name at the pathology level
    (that check happens in ``PanelGenerator``).
    """

    def __init__(
        self,
        period: int,
        target: str,
        kind: Literal["mean_shift", "variance_shift", "coefficient_shift"],
        magnitude: float,
        coefficient_target: str | None = None,
    ) -> None:
        if kind not in _STRUCTURAL_BREAK_KINDS:
            raise ValueError(f"Unsupported kind {kind!r}; supported: {_STRUCTURAL_BREAK_KINDS}")
        if kind == "coefficient_shift" and coefficient_target is None:
            raise ValueError("coefficient_target is required when kind='coefficient_shift'")
        if period < 0:
            raise ValueError(f"period must be >= 0, got {period}")

        self.period = period
        self.target = target
        self.kind = kind
        self.magnitude = magnitude
        self.coefficient_target = coefficient_target

    def apply_to_mean(
        self,
        mean_outcome: np.ndarray,
        periods: np.ndarray,
        design: pd.DataFrame | None,
        coefficients: dict[str, float] | None,
    ) -> np.ndarray:
        """Return a modified mean-outcome array reflecting this break.

        `design`/`coefficients` are only used for `kind="coefficient_shift"`
        and may be None otherwise.
        """
        post = periods >= self.period
        if self.kind == "mean_shift":
            result = mean_outcome.copy()
            result[post] = result[post] + self.magnitude
            return result

        if self.kind == "coefficient_shift":
            if design is None or coefficients is None:
                raise ValueError(
                    "coefficient_shift requires a formula-specified outcome "
                    "(design matrix and coefficients); it is not supported "
                    "with an fn=-specified outcome."
                )
            assert self.coefficient_target is not None
            if self.coefficient_target not in design.columns:
                raise ValueError(
                    f"coefficient_target {self.coefficient_target!r} is not a column of the "
                    f"outcome design matrix; available columns: {list(design.columns)}"
                )
            col = design[self.coefficient_target].to_numpy()
            old_coef = coefficients.get(self.coefficient_target, 0.0)
            delta_coef = self.magnitude - old_coef
            result = mean_outcome.copy()
            result[post] = result[post] + delta_coef * col[post]
            return result

        # variance_shift does not alter the mean; handled separately via
        # apply_to_noise_std.
        return mean_outcome

    def apply_to_noise_std(self, base_noise_std: float, periods: np.ndarray) -> np.ndarray:
        """Return a per-row noise std array reflecting this break (only
        meaningful for kind="variance_shift"; otherwise returns the
        unmodified base std broadcast to `periods`' shape)."""
        std = np.full(periods.shape, base_noise_std, dtype=float)
        if self.kind == "variance_shift":
            post = periods >= self.period
            std[post] = std[post] * self.magnitude
        return std

    def ground_truth_contribution(self) -> dict:
        return {
            "break_points": [
                BreakInfo(period=self.period, target=self.target, kind=self.kind, magnitude=self.magnitude)
            ]
        }

    def conflicts_with(self, other: Pathology) -> str | None:
        if isinstance(other, StructuralBreak) and other.period == self.period and other.target == self.target:
            return (
                f"Two structural breaks target the same outcome {self.target!r} at the same "
                f"period {self.period}; their effects will stack, which may not be intended."
            )
        return None


class Confounder(Pathology):
    """Injects an unobserved (or observed) variable that confounds a feature
    and an outcome, inducing a spurious relationship. Implemented in a
    subsequent commit."""

    def __init__(self, feature: str, outcome: str, strength: float, observed: bool = False) -> None:
        self.feature = feature
        self.outcome = outcome
        self.strength = strength
        self.observed = observed

    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError


class SelectionBias(Pathology):
    """Applies non-random sample selection according to a boolean rule,
    evaluated via a constrained pandas.eval (see CONTRIBUTING.md).
    Implemented in a subsequent commit."""

    def __init__(self, rule: str, drop_prob: float) -> None:
        self.rule = rule
        self.drop_prob = drop_prob

    def ground_truth_contribution(self) -> dict:
        raise NotImplementedError


def validate_combo(pathologies: list[Pathology]) -> list[str]:
    """Check a list of pathologies for likely conflicts. Returns a list of
    warning messages (empty if none). Never raises — v1 policy is permissive."""
    warnings: list[str] = []
    for i, p in enumerate(pathologies):
        for other in pathologies[i + 1 :]:
            msg = p.conflicts_with(other)
            if msg is not None:
                warnings.append(msg)
    return warnings
