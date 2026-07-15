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
from typing import Literal, Optional

import numpy as np
import pandas as pd

from spuriosity.ground_truth import (
    BreakInfo,
    EndogeneityInfo,
    HeteroskedasticityInfo,
    MeasurementErrorInfo,
    MulticollinearityInfo,
    SelectionInfo,
    UnitRootInfo,
)

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


class Heteroskedasticity(Pathology):
    """Makes the outcome's noise standard deviation vary as a function of
    `feature`, rather than being constant, via a `pandas.eval`-evaluated
    `formula` in terms of `feature` (e.g. ``"1 + 0.5*x1**2"``).

    Mechanism: at each row, the outcome noise std is
    ``base_noise_std * eval(formula, feature=value)`` instead of a flat
    ``base_noise_std`` (the `noise_std` supplied to
    `PanelGenerator.set_outcome`). `formula` should evaluate to a
    non-negative multiplier; a warning-worthy but not hard-blocked case is
    a formula that goes negative for some observed range of `feature`
    (clamped to 0 at generation time, which would silently zero out noise
    for those rows -- callers should choose formulas that stay positive
    over the feature's realistic range, e.g. squared or exponential
    forms rather than formulas that can cross zero).

    This does not bias OLS point estimates (heteroskedasticity is a
    violation of the constant-variance assumption, not of exogeneity), but
    it invalidates naive (non-robust) standard errors -- the textbook
    consequence, and the property this pathology exists to let users
    verify their pipeline actually checks for (e.g. via White/HC3 robust
    SEs or a Breusch-Pagan test) rather than trusting default OLS SEs.
    """

    def __init__(self, feature: str, formula: str) -> None:
        self.feature = feature
        self.formula = formula

    def compute_noise_multiplier(self, feature_values: np.ndarray) -> np.ndarray:
        """Evaluate `formula` against `feature_values`, returning the
        per-row noise standard deviation multiplier. Negative results are
        clamped to 0 (see class docstring)."""
        try:
            result = pd.eval(
                self.formula,
                local_dict={self.feature: pd.Series(feature_values)},
                global_dict={},
                engine="python",
            )
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate heteroskedasticity formula {self.formula!r} for feature "
                f"{self.feature!r}: {e}"
            ) from e

        arr = np.asarray(result, dtype=float)
        if arr.shape == ():
            arr = np.full(feature_values.shape, float(arr))
        return np.clip(arr, a_min=0.0, a_max=None)

    def ground_truth_contribution(self) -> dict:
        return {
            "heteroskedasticity": [
                HeteroskedasticityInfo(feature=self.feature, formula=self.formula)
            ],
        }


class Multicollinearity(Pathology):
    """Generates `feature` as a near-linear function of an existing
    `correlated_with` variable plus independent noise, calibrated so the
    two columns have Pearson correlation approximately `correlation`
    (0 <= correlation < 1; use values close to 1, e.g. 0.9-0.99, to model
    realistic near-collinearity -- correlation=1.0 is disallowed since
    perfect collinearity makes OLS undefined rather than merely unstable,
    which is a degenerate edge case rather than the "high but estimable
    VIF" scenario this pathology is meant to model).

    Mechanism: given standardized `correlated_with` (call it `z`, i.e.
    mean 0, unit variance over the generated sample) and target
    correlation `rho`, the new feature is constructed as
    ``feature = rho * z + sqrt(1 - rho**2) * epsilon`` where `epsilon` is
    independent standard normal noise. This construction gives
    `feature` unit variance and, in expectation, exactly `rho` correlation
    with `correlated_with` -- the realized sample correlation will be close
    to but not bit-for-bit exactly `rho` (finite-sample noise), converging
    to `rho` as sample size grows.

    For two features, the closed-form prediction this pathology exists to
    let users verify is `VIF = 1 / (1 - rho**2)` for the collinear feature
    (regressed on just `correlated_with`); `StressTest`/manual checks can
    compare a fitted VIF against this.
    """

    def __init__(self, feature: str, correlated_with: str, correlation: float) -> None:
        if not (0.0 <= correlation < 1.0):
            raise ValueError(
                f"correlation must be in [0, 1) -- got {correlation}. "
                "correlation=1.0 (perfect collinearity) is disallowed; OLS is undefined "
                "in that degenerate case rather than merely high-variance."
            )
        self.feature = feature
        self.correlated_with = correlated_with
        self.correlation = correlation

    def generate_feature(
        self,
        correlated_with_values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate the new collinear feature's values, standardizing
        `correlated_with_values` internally so the target correlation is
        achieved regardless of that column's own scale."""
        z = correlated_with_values
        std = z.std()
        if std == 0:
            raise ValueError(
                f"Cannot generate a feature correlated with {self.correlated_with!r}: "
                "that column has zero variance in the generated data."
            )
        z_standardized = (z - z.mean()) / std
        epsilon = rng.normal(loc=0.0, scale=1.0, size=z.shape[0])
        rho = self.correlation
        new_feature: np.ndarray = rho * z_standardized + np.sqrt(1 - rho**2) * epsilon
        return new_feature

    def ground_truth_contribution(self) -> dict:
        return {
            "multicollinearity": [
                MulticollinearityInfo(
                    feature=self.feature,
                    correlated_with=self.correlated_with,
                    target_correlation=self.correlation,
                )
            ],
        }


class MeasurementError(Pathology):
    """Injects classical measurement error into `feature`: the value that
    ends up in the generated DataFrame is the true (already-drawn) value
    plus independent Gaussian noise with standard deviation `noise_std`,
    i.e. ``observed = true_value + N(0, noise_std**2)``.

    Critically, the outcome is generated from the TRUE (pre-error) values
    -- only the feature column visible in the final DataFrame is
    corrupted, mirroring the realistic scenario where the underlying
    construct that actually drives the outcome is measured imperfectly.
    This is the mechanistic opposite of `Confounder` (which injects a
    latent variable that the outcome also depends on): here, noise is
    added purely to the *measurement* of an existing regressor, with no
    effect on the outcome except through whatever the true (unobserved)
    value would have contributed.

    The classical result this pathology exists to let users verify: a
    naive regression of the outcome on the *noisy* observed feature has
    its coefficient attenuated toward zero by the reliability ratio
    ``Var(true) / (Var(true) + noise_std**2)`` -- unlike confounding
    (which inflates a coefficient) or a simple omitted variable, this
    bias always pulls the estimate toward zero, never away from it.
    """

    def __init__(self, feature: str, noise_std: float) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self.feature = feature
        self.noise_std = noise_std
        self._realized_reliability_ratio: Optional[float] = None

    def apply(self, true_values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the noisy observed values, and record the realized
        reliability ratio (based on the true values' actual sample
        variance) for ground-truth bookkeeping via
        `ground_truth_contribution()`, which must be called only after
        this method."""
        true_variance = float(np.var(true_values))
        if self.noise_std == 0:
            self._realized_reliability_ratio = 1.0
        else:
            self._realized_reliability_ratio = true_variance / (true_variance + self.noise_std**2)

        noise = rng.normal(loc=0.0, scale=self.noise_std, size=true_values.shape[0])
        observed: np.ndarray = true_values + noise
        return observed

    def ground_truth_contribution(self) -> dict:
        if self._realized_reliability_ratio is None:
            raise RuntimeError(
                "ground_truth_contribution() called before apply(); the realized reliability "
                "ratio depends on the true values' sample variance, which is only known after "
                "apply() has run."
            )
        return {
            "measurement_error": [
                MeasurementErrorInfo(
                    feature=self.feature,
                    noise_std=self.noise_std,
                    reliability_ratio=self._realized_reliability_ratio,
                )
            ],
        }


class Endogeneity(Pathology):
    """Makes `feature` endogenous (correlated with the outcome's error
    term) via a shared latent variable `u`, while also generating a valid
    exogenous `instrument` column that can recover the true coefficient
    via 2SLS/IV instead of naive OLS.

    Mechanism, in generation order:

        instrument ~ N(0, 1)                           [new column]
        u ~ N(0, 1)                                     [latent, not exposed]
        feature = instrument_strength * instrument + endogeneity_strength * u
                  + N(0, first_stage_noise_std**2)       [replaces feature's prior values]
        outcome_mean += endogeneity_strength * u         [added on top of the outcome DGP]

    `u` leaking into both `feature` and the outcome is what makes `feature`
    endogenous: naive OLS of outcome on `feature` is biased, because
    `Cov(feature, u) != 0` and `u` also drives the outcome. `instrument`
    is exogenous by construction (independent of `u`), so 2SLS using
    `instrument` recovers the true coefficient.

    `instrument_strength` is the first-stage coefficient -- how strongly
    `instrument` drives `feature`. Deliberately setting this low (e.g.
    0.05-0.1 relative to `endogeneity_strength`) generates data with a
    *weak* instrument: 2SLS becomes both biased and highly imprecise, with
    a low first-stage F-statistic (the standard weak-instrument
    diagnostic) -- this is the "does my IV strategy survive a weak
    instrument" stress test this pathology exists to support, not just
    "does IV work when everything is textbook-strong."

    `feature` must already be declared via `add_variable` (this pathology
    replaces its values, mirroring `Confounder`'s convention, not
    `Multicollinearity`'s "creates a new column" convention).
    `instrument` must NOT already be declared -- it is created fresh.
    """

    def __init__(
        self,
        feature: str,
        instrument: str,
        instrument_strength: float,
        endogeneity_strength: float,
        first_stage_noise_std: float = 0.5,
    ) -> None:
        if first_stage_noise_std < 0:
            raise ValueError(f"first_stage_noise_std must be >= 0, got {first_stage_noise_std}")
        self.feature = feature
        self.instrument = instrument
        self.instrument_strength = instrument_strength
        self.endogeneity_strength = endogeneity_strength
        self.first_stage_noise_std = first_stage_noise_std
        self._realized_f_stat: Optional[float] = None

    def generate(
        self, n_rows: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate (instrument_values, feature_values, outcome_contribution).

        `outcome_contribution` is `endogeneity_strength * u`, to be added
        to the outcome mean by the caller (mirroring how `Confounder`
        returns its outcome contribution separately rather than mutating
        the outcome directly).
        """
        instrument_values = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        u = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        first_stage_noise = rng.normal(loc=0.0, scale=self.first_stage_noise_std, size=n_rows)

        feature_values = (
            self.instrument_strength * instrument_values
            + self.endogeneity_strength * u
            + first_stage_noise
        )
        outcome_contribution = self.endogeneity_strength * u

        # Realized first-stage F-stat: F-test of instrument_strength == 0
        # in a simple OLS of feature on instrument, computed on the actual
        # generated sample (not a theoretical value) -- the standard
        # weak-instrument diagnostic.
        self._realized_f_stat = _first_stage_f_stat(instrument_values, feature_values)

        return instrument_values, feature_values, outcome_contribution

    def ground_truth_contribution(self) -> dict:
        return {
            "endogeneity": [
                EndogeneityInfo(
                    feature=self.feature,
                    instrument=self.instrument,
                    instrument_strength=self.instrument_strength,
                    endogeneity_strength=self.endogeneity_strength,
                    realized_first_stage_f_stat=self._realized_f_stat,
                )
            ],
        }


def _first_stage_f_stat(instrument_values: np.ndarray, feature_values: np.ndarray) -> float:
    """F-statistic for the null that the instrument's coefficient is 0, in
    a simple OLS regression of feature_values on instrument_values (plus
    intercept). Computed directly via the standard F = t**2 relationship
    for a single-regressor OLS, to avoid a statsmodels dependency inside
    pathologies.py for what is otherwise a pure numpy module.
    """
    n = len(instrument_values)
    x = instrument_values
    y = feature_values
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - y_mean))
    beta = sxy / sxx
    intercept = y_mean - beta * x_mean
    residuals = y - (intercept + beta * x)
    rss = np.sum(residuals**2)
    dof = n - 2
    se_beta = np.sqrt(rss / dof / sxx)
    t_stat = beta / se_beta
    f_stat: float = float(t_stat**2)
    return f_stat


class UnitRoot(Pathology):
    """Converts `feature` from i.i.d. draws into a random walk (with
    optional drift), making it nonstationary: variance grows with the
    time index rather than being constant, which breaks standard OLS
    inference and can produce "spurious regression" -- an apparently
    significant relationship between two variables that are in fact
    causally unrelated, purely because both wander/trend over time.

    Mechanism: within each entity's own time series (never crossing
    entity boundaries in a panel), `feature`'s already-drawn i.i.d. values
    are treated as increments and cumulatively summed:

        feature_t = feature_0 + sum_{s=1}^{t} (increment_s + drift)

    i.e. the random walk is built from the SAME underlying draws
    `add_variable` already produced for `feature` (reusing them as
    increments rather than drawing fresh noise), so the only thing this
    pathology changes is the cumulative-sum transformation and the
    optional additive `drift` term at each step.

    `drift`, if nonzero, adds a deterministic trend on top of the random
    walk (a "random walk with drift") -- still nonstationary, but with a
    systematic directional component in addition to the wandering.

    This pathology operates on the full panel structure (it needs each
    entity's own time ordering to build a per-entity cumulative sum), so
    `apply_to_panel` takes the whole DataFrame rather than a flat array,
    unlike most other pathologies in this module.
    """

    def __init__(self, feature: str, drift: float = 0.0) -> None:
        self.feature = feature
        self.drift = drift

    def apply_to_panel(
        self, df: pd.DataFrame, entity_col: str = "entity_id", period_col: str = "period"
    ) -> np.ndarray:
        """Return the feature column converted to a per-entity random walk
        with drift, preserving the DataFrame's row order. `df` must
        already contain `self.feature`, `entity_col`, and `period_col`,
        and rows are assumed sorted by (entity, period) within each
        entity's block -- true for any DataFrame produced by
        `PanelGenerator.generate()`.
        """
        if self.feature not in df.columns:
            raise ValueError(
                f"UnitRoot references feature {self.feature!r}, which is not present in the data."
            )
        increments = df[self.feature].to_numpy() + self.drift
        # groupby.cumsum() resets the cumulative sum at each entity
        # boundary, which is exactly the per-entity random walk semantics
        # this pathology needs -- verified against a hand-built loop
        # during development.
        walk: np.ndarray = (
            pd.Series(increments, index=df.index).groupby(df[entity_col]).cumsum().to_numpy()
        )
        return walk

    def ground_truth_contribution(self) -> dict:
        return {
            "unit_root": [UnitRootInfo(feature=self.feature, drift=self.drift)],
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
