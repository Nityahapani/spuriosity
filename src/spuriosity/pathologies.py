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

from spuriosity.ground_truth import BreakInfo, SelectionInfo

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
    """Injects a latent variable `U ~ N(0, 1)` that causally affects both
    `feature` and `outcome`, inducing omitted-variable bias if `U` is not
    controlled for.

    Mechanism (added on top of whatever DGP `feature` and `outcome` already
    have):

        feature += strength * U
        outcome_mean += strength * U

    With `Var(U) = 1`, this induces a precise, checkable omitted-variable
    bias in a naive regression of `outcome` on `feature` alone: the naive
    coefficient on `feature` is biased upward by

        bias = strength**2 / (1 + strength**2)

    relative to the true coefficient. This holds because
    `Cov(feature, U) = strength`, `Var(feature) = 1 + strength**2`
    (assuming the feature's own base variance is 1; see note below), and
    the standard omitted-variable-bias formula gives
    `bias = Cov(feature, U) * Cov(U, outcome | feature) / Var(feature)`,
    which simplifies to the expression above when `U` enters both `feature`
    and `outcome` linearly with the same coefficient `strength`.

    Note: the formula above assumes `feature`'s own (pre-confounding)
    variance is 1 (the `add_variable` default for `dist="normal"`). If
    `feature` was declared with a different `std`, the exact bias differs;
    `ground_truth_contribution()` still reports `strength` so the true
    mechanism is always available even if the closed-form bias needs
    adjusting for a given `std`.

    If `observed=True`, `U` is added to the generated DataFrame as a
    visible column named `f"_confounder_{feature}"`, letting a pipeline
    that actually controls for it recover the true coefficient. If
    `observed=False` (default), `U` influences the data but is not
    included in the output -- the realistic "hidden confounder" case.

    Caution: if `feature` is a binary treatment indicator (declared via
    `add_treatment`), confounding it additively turns it into a continuous
    variable (no longer strictly 0/1), which silently breaks pipelines that
    assume a binary treatment (e.g. DiD estimators, propensity models).
    This is intentional -- it models continuous-dose confounding -- but is
    usually not what you want when testing a binary-treatment estimator.
    Prefer confounding a covariate rather than the treatment itself unless
    you specifically want this effect.
    """

    def __init__(self, feature: str, outcome: str, strength: float, observed: bool = False) -> None:
        self.feature = feature
        self.outcome = outcome
        self.strength = strength
        self.observed = observed

    def draw_and_apply(
        self,
        feature_values: np.ndarray,
        outcome_mean: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw U and apply it to both feature and outcome mean.

        Returns (new_feature_values, new_outcome_mean, U) so the caller can
        optionally expose U as a visible column when `observed=True`.
        """
        u: np.ndarray = np.asarray(rng.normal(loc=0.0, scale=1.0, size=feature_values.shape[0]))
        new_feature: np.ndarray = feature_values + self.strength * u
        new_outcome_mean: np.ndarray = outcome_mean + self.strength * u
        return new_feature, new_outcome_mean, u

    def predicted_naive_bias(self, feature_std: float = 1.0) -> float:
        """Predicted bias in a naive OLS coefficient on `feature` (from a
        regression of `outcome` on `feature` alone, omitting `U`), assuming
        `feature`'s own pre-confounding variance is `feature_std ** 2`.

        Derivation: with feature = feature_base + strength*U where
        Var(feature_base) = feature_std**2 and Var(U) = 1,
        Cov(feature, U) = strength, Var(feature) = feature_std**2 +
        strength**2. Since outcome_mean also includes + strength*U,
        Cov(U, outcome | feature-induced-part) contributes a bias term of
        strength * Cov(feature, U) / Var(feature) = strength**2 /
        (feature_std**2 + strength**2).
        """
        return self.strength**2 / (feature_std**2 + self.strength**2)

    def ground_truth_contribution(self) -> dict:
        return {
            "confounding_strength": {self.feature: self.strength},
        }


class SelectionBias(Pathology):
    """Applies non-random sample selection: rows matching `rule` are dropped
    with probability `drop_prob` (not deterministically -- this lets the
    severity of selection be dialed in rather than being all-or-nothing).

    `rule` is a boolean expression evaluated via `pandas.eval` against the
    generated DataFrame's columns (including the outcome, so
    outcome-dependent selection / survivorship bias can be modeled), using
    an explicit, minimal namespace -- never the ambient globals/locals of
    the calling code. See CONTRIBUTING.md for the full security stance.

    Selection is applied after the full DataFrame (including the outcome)
    has been generated, and results in rows being physically removed from
    the output -- mirroring what a real dataset with non-random
    missingness actually looks like, rather than exposing a "selected"
    flag column that a naive pipeline could accidentally ignore.
    """

    def __init__(self, rule: str, drop_prob: float) -> None:
        if not (0.0 <= drop_prob <= 1.0):
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        self.rule = rule
        self.drop_prob = drop_prob

    def compute_mask_to_drop(
        self,
        data: pd.DataFrame,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Evaluate the rule against `data` and return a boolean array of
        which rows to drop (True = drop).

        Raises ValueError if `rule` does not evaluate to a boolean-typed
        result (e.g. an arithmetic expression rather than a comparison),
        since a non-boolean rule almost certainly indicates a mistake
        rather than an intended selection mechanism.
        """
        local_dict = {col: data[col] for col in data.columns}
        try:
            matches = pd.eval(self.rule, local_dict=local_dict, global_dict={}, engine="python")
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate selection rule {self.rule!r}: {e}. Rules must be boolean "
                "expressions over columns present in the generated DataFrame, e.g. 'x1 > 1.5'."
            ) from e

        matches_arr = np.asarray(matches)
        if matches_arr.dtype != bool:
            raise ValueError(
                f"Selection rule {self.rule!r} did not evaluate to a boolean result "
                f"(got dtype {matches_arr.dtype}); rules must be boolean expressions, "
                "e.g. 'x1 > 1.5', not arithmetic expressions."
            )

        random_draw = rng.random(size=len(data))
        drop_mask: np.ndarray = matches_arr & (random_draw < self.drop_prob)
        return drop_mask

    def ground_truth_contribution(self) -> dict:
        return {
            "selection_mechanism": SelectionInfo(rule=self.rule, drop_prob=self.drop_prob),
        }


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
