"""
GroundTruth — the frozen record of a PanelGenerator's true data-generating
process, returned alongside every generated DataFrame.

Reproducibility contract: same seed + same pinned spuriosity/numpy versions
produces a byte-identical DataFrame. No cross-version guarantee — see
docs/design_spec.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BreakInfo:
    """Record of a single injected structural break."""

    period: int
    target: str
    kind: str
    magnitude: float

    def __repr__(self) -> str:
        # Lead with the kind, since that's what you scan for first when debugging
        return f"BreakInfo(kind={self.kind!r}, period={self.period}, target={self.target!r}, magnitude={self.magnitude})"


@dataclass(frozen=True)
class SelectionInfo:
    """Record of the injected selection-bias mechanism."""

    rule: str
    drop_prob: float

    def __repr__(self) -> str:
        return f"SelectionInfo(rule={self.rule!r}, drop_prob={self.drop_prob})"


@dataclass(frozen=True)
class HeteroskedasticityInfo:
    """Record of a single injected heteroskedasticity mechanism: the
    outcome's noise standard deviation is scaled by `formula` (evaluated
    in terms of `feature`) rather than held constant."""

    feature: str
    formula: str

    def __repr__(self) -> str:
        return f"HeteroskedasticityInfo(feature={self.feature!r}, formula={self.formula!r})"


@dataclass(frozen=True)
class MulticollinearityInfo:
    """Record of a single injected multicollinearity mechanism: `feature`
    was generated as a near-linear function of `correlated_with`, targeting
    Pearson correlation `target_correlation`."""

    feature: str
    correlated_with: str
    target_correlation: float

    def __repr__(self) -> str:
        return (
            f"MulticollinearityInfo(feature={self.feature!r}, "
            f"correlated_with={self.correlated_with!r}, target_correlation={self.target_correlation})"
        )


@dataclass(frozen=True)
class MeasurementErrorInfo:
    """Record of a single injected classical measurement error mechanism:
    `feature`'s observed values are `true_value + noise`, where noise has
    standard deviation `noise_std`. `reliability_ratio` is the closed-form
    predicted attenuation factor for a naive regression coefficient on the
    noisy observed feature, `Var(true) / (Var(true) + noise_std**2)`,
    computed from the feature's *realized* variance in the generated
    sample (not a theoretical value), so it reflects the actual data.
    """

    feature: str
    noise_std: float
    reliability_ratio: float

    def __repr__(self) -> str:
        return (
            f"MeasurementErrorInfo(feature={self.feature!r}, noise_std={self.noise_std}, "
            f"reliability_ratio={self.reliability_ratio:.4f})"
        )


@dataclass(frozen=True)
class EndogeneityInfo:
    """Record of a single injected endogeneity mechanism: `feature` is
    entangled with the outcome's error term via a shared latent variable,
    and `instrument` is the exogenous instrument that can be used to
    recover the true coefficient via 2SLS/IV instead of naive OLS.

    `instrument_strength` is the first-stage coefficient (how strongly the
    instrument drives the endogenous feature); `endogeneity_strength` is
    how strongly the shared latent error leaks into both the feature and
    the outcome (i.e. the severity of the OLS bias this pathology
    induces). `realized_first_stage_f_stat`, if computed (see
    `Endogeneity.ground_truth_contribution`), is the standard
  weak-instrument diagnostic (first-stage F-statistic on the instrument's
    coefficient) for the actual generated sample -- values below the
    classic Stock-Yogo rule-of-thumb of 10 indicate a weak instrument for
    that specific dataset.
    """

    feature: str
    instrument: str
    instrument_strength: float
    endogeneity_strength: float
    realized_first_stage_f_stat: Optional[float] = None

    def __repr__(self) -> str:
        f_stat_str = (
            f"{self.realized_first_stage_f_stat:.1f}"
            if self.realized_first_stage_f_stat is not None
            else "None"
        )
        return (
            f"EndogeneityInfo(feature={self.feature!r}, instrument={self.instrument!r}, "
            f"instrument_strength={self.instrument_strength}, "
            f"endogeneity_strength={self.endogeneity_strength}, "
            f"realized_first_stage_f_stat={f_stat_str})"
        )


@dataclass(frozen=True)
class UnitRootInfo:
    """Record of a single injected unit-root (random walk) mechanism:
    `feature`'s values were converted from i.i.d. draws into a random walk
    (cumulative sum of increments), optionally with a constant `drift`
    added at each step. This makes `feature` nonstationary -- its variance
    grows with the time index rather than being constant -- which breaks
    standard OLS inference and can produce "spurious regression": an
    apparently significant relationship between two variables that are in
    fact causally unrelated, purely because both are trending/wandering
    random walks. See `spuriosity.pathologies.UnitRoot` for the mechanism
    and the verified spurious-regression false-positive-rate inflation
    this pathology reproduces.
    """

    feature: str
    drift: float

    def __repr__(self) -> str:
        return f"UnitRootInfo(feature={self.feature!r}, drift={self.drift})"


@dataclass(frozen=True)
class GroundTruth:
    """The true data-generating process behind a generated panel dataset.

    `true_cate`, if set, is a callable excluded from serialization (it isn't
    JSON-representable) — `to_dict()`/`to_json()` note its presence via
    `has_true_cate` instead of attempting to serialize the function itself.

    Reproducibility contract: same `seed` + same pinned `spuriosity` and
    `numpy` versions produces a byte-identical generated DataFrame. No
    cross-version guarantee is made — see docs/design_spec.md.
    """

    true_coefficients: dict[str, float]
    break_points: list[BreakInfo] = field(default_factory=list)
    confounding_strength: dict[str, float] = field(default_factory=dict)
    true_cate: Optional[Callable[[float], float]] = None
    selection_mechanism: Optional[SelectionInfo] = None
    heteroskedasticity: list[HeteroskedasticityInfo] = field(default_factory=list)
    multicollinearity: list[MulticollinearityInfo] = field(default_factory=list)
    measurement_error: list[MeasurementErrorInfo] = field(default_factory=list)
    endogeneity: list[EndogeneityInfo] = field(default_factory=list)
    unit_root: list[UnitRootInfo] = field(default_factory=list)
    treatment_effect_ate: Optional[float] = None
    spuriosity_version: str = ""
    numpy_version: str = ""
    seed: int = 0

    def __repr__(self) -> str:
        """Compact, debugging-friendly summary of the ground truth.

        Designed for notebooks: one screen of text, every populated field
        visible at a glance, no truncation of the coefficient dict
        (these are usually 2–6 entries).
        """
        parts: list[str] = [f"GroundTruth(seed={self.seed}, spuriosity={self.spuriosity_version!r})"]

        if self.true_coefficients:
            coefs = ", ".join(f"{k!r}: {v}" for k, v in self.true_coefficients.items())
            parts.append(f"  true_coefficients: {{{coefs}}}")
        else:
            parts.append("  true_coefficients: <empty>")

        if self.break_points:
            parts.append(f"  break_points: {len(self.break_points)} ({[b.kind for b in self.break_points]})")
        if self.confounding_strength:
            parts.append(f"  confounding_strength: {self.confounding_strength}")
        if self.selection_mechanism is not None:
            sm = self.selection_mechanism
            parts.append(f"  selection_mechanism: rule={sm.rule!r}, drop_prob={sm.drop_prob}")
        if self.heteroskedasticity:
            parts.append(
                f"  heteroskedasticity: {[h.feature for h in self.heteroskedasticity]}"
            )
        if self.multicollinearity:
            parts.append(
                f"  multicollinearity: {[(m.feature, m.correlated_with) for m in self.multicollinearity]}"
            )
        if self.measurement_error:
            parts.append(
                f"  measurement_error: {[(m.feature, round(m.reliability_ratio, 3)) for m in self.measurement_error]}"
            )
        if self.endogeneity:
            parts.append(
                f"  endogeneity: {[(e.feature, e.instrument) for e in self.endogeneity]}"
            )
        if self.unit_root:
            parts.append(f"  unit_root: {[u.feature for u in self.unit_root]}")
        if self.treatment_effect_ate is not None:
            parts.append(f"  treatment_effect_ate: {self.treatment_effect_ate:.4f}")
        parts.append(f"  has_true_cate: {self.true_cate is not None}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict. `true_cate`, being a callable, is not
        included directly; its presence is flagged via `has_true_cate`."""
        d: dict[str, Any] = {
            "true_coefficients": dict(self.true_coefficients),
            "break_points": [asdict(b) for b in self.break_points],
            "confounding_strength": dict(self.confounding_strength),
            "has_true_cate": self.true_cate is not None,
            "selection_mechanism": (
                asdict(self.selection_mechanism) if self.selection_mechanism is not None else None
            ),
            "heteroskedasticity": [asdict(h) for h in self.heteroskedasticity],
            "multicollinearity": [asdict(m) for m in self.multicollinearity],
            "measurement_error": [asdict(m) for m in self.measurement_error],
            "endogeneity": [asdict(e) for e in self.endogeneity],
            "unit_root": [asdict(u) for u in self.unit_root],
            "treatment_effect_ate": self.treatment_effect_ate,
            "spuriosity_version": self.spuriosity_version,
            "numpy_version": self.numpy_version,
            "seed": self.seed,
        }
        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string via `to_dict()`."""
        return json.dumps(self.to_dict(), indent=indent)
