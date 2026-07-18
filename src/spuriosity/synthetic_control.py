"""
Synthetic Control Method (Abadie, Diamond & Hainmueller 2010) with
placebo-in-space inference (Abadie, Diamond & Hainmueller 2010, 2015).

Given a single treated unit and a pool of untreated "donor" units observed
over the same time periods, constructs a "synthetic control" -- a convex
combination (nonnegative weights summing to 1) of donor units chosen to
match the treated unit's pre-treatment outcome trajectory as closely as
possible. The estimated treatment effect at each post-treatment period is
the gap between the treated unit's actual outcome and its synthetic
control's outcome.

Because there is only one treated unit, standard parametric standard
errors don't apply. The standard inferential approach in this literature
is "placebo-in-space": rerun the identical procedure treating each donor
unit as if it were the treated unit (using the real treated unit and the
remaining donors as its donor pool), producing a distribution of placebo
effects under the null of no true effect. The real unit's effect is then
compared against this placebo distribution.

Caveat confirmed during development: because each placebo unit's donor
pool includes the REAL treated unit (whose post-period values carry the
true effect, if any), a donor with substantial weight on the treated unit
can itself produce a large apparent "placebo effect" when compensating
for that shift in its own fit -- this can occasionally make a placebo
unit's effect exceed the real treated unit's in magnitude, especially
with a small donor pool. This is a genuine, known limitation of
placebo-in-space inference (not a bug), and means the real effect is not
guaranteed to rank as the single most extreme unit even when it is
genuine -- the resulting p-value is best interpreted as "how unusual is
this effect relative to the placebo distribution," not as a guarantee
that a true effect will always achieve the minimum possible p-value.
Larger, more diverse donor pools make this less of a concern.

This module is deliberately kept separate from reference.py: unlike the
thin fit/predict wrappers there, SCM has real internal structure (a
constrained optimization step, a placebo-inference procedure with its own
result object) that benefits from a dedicated module rather than a pair
of loose functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SyntheticControlResult:
    """Result of fitting a synthetic control for a single treated unit.

    `weights` maps each donor unit's identifier to its fitted weight
    (nonnegative, summing to ~1). `synthetic_outcome` is the synthetic
    control's fitted outcome trajectory over ALL periods (pre and post);
    `treated_outcome` is the real treated unit's observed trajectory over
    the same periods, for direct comparison/plotting.

    `effect_by_period` is the post-treatment gap (treated - synthetic) at
    each post-treatment period; `average_effect` is its mean.

    `pre_period_fit_rmse` is the root-mean-squared pre-treatment gap
    between the treated unit and its synthetic control -- a standard SCM
    diagnostic: a large pre-period RMSE means the synthetic control didn't
    actually match the treated unit's pre-treatment trajectory well, which
    undermines confidence in the post-treatment gap being attributable to
    the treatment rather than poor fit.
    """

    treated_unit: object
    weights: dict[object, float]
    periods: np.ndarray
    treatment_period: object
    treated_outcome: np.ndarray
    synthetic_outcome: np.ndarray
    effect_by_period: np.ndarray
    average_effect: float
    pre_period_fit_rmse: float
    placebo_effects: Optional[dict[object, np.ndarray]] = field(default=None)
    placebo_p_value: Optional[float] = field(default=None)

    def summary(self) -> None:
        print(f"SyntheticControlResult: treated_unit={self.treated_unit!r}")
        print(f"  Pre-period fit RMSE: {self.pre_period_fit_rmse:.4f}")
        print(f"  Average post-treatment effect: {self.average_effect:.4f}")
        top_weights = sorted(self.weights.items(), key=lambda kv: -kv[1])[:5]
        weight_str = ", ".join(f"{unit!r}={w:.3f}" for unit, w in top_weights if w > 1e-4)
        print(f"  Top donor weights: {weight_str}")
        if self.placebo_p_value is not None:
            print(f"  Placebo-in-space p-value: {self.placebo_p_value:.4f}")


def _fit_weights(target_pre: np.ndarray, donor_pool_pre: np.ndarray) -> np.ndarray:
    """Fit nonnegative, sum-to-1 weights minimizing the squared gap
    between `target_pre` and `weights @ donor_pool_pre`, via SLSQP.

    `donor_pool_pre` has shape (n_donors, n_pre_periods).
    """
    n_donors = donor_pool_pre.shape[0]

    def loss(w: np.ndarray) -> float:
        return float(np.sum((target_pre - w @ donor_pool_pre) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    w0 = np.ones(n_donors) / n_donors
    result = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights: np.ndarray = result.x
    # Numerical cleanup: SLSQP can leave tiny negative/over-1 residuals
    # from floating point noise; clip and renormalize so weights are
    # exactly valid (nonnegative, sum to 1) rather than approximately so.
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    return weights


def synthetic_control_fit(
    data: pd.DataFrame,
    outcome: str,
    entity_col: str,
    period_col: str,
    treated_unit: object,
    treatment_period: object,
    donor_units: Optional[list] = None,
    run_placebo_inference: bool = True,
) -> SyntheticControlResult:
    """Fit a synthetic control for `treated_unit` using the other units in
    `data` (or `donor_units`, if explicitly given) as the donor pool.

    `treatment_period` is compared against `data[period_col]` via `>=` to
    split pre/post periods -- periods before `treatment_period` are used
    to fit the weights; periods from `treatment_period` onward are the
    post-treatment window where the effect is estimated.

    If `run_placebo_inference=True` (default), also runs placebo-in-space
    inference: refits the identical procedure treating each donor unit as
    the "treated" unit (using the real treated unit and remaining donors
    as its pool), producing `placebo_effects` (each placebo unit's
    average post-treatment effect trajectory) and a rank-based
    `placebo_p_value` -- the fraction of units (including the real one)
    whose |average effect| is >= the real treated unit's |average
    effect|. Note this p-value's granularity is limited by the number of
    donors: with `k` donors, the smallest achievable p-value is `1/(k+1)`
    -- e.g. 5 donors can never yield a p-value below ~0.167, a real
    limitation of placebo-in-space inference with a small donor pool that
    should be interpreted with this floor in mind, not treated as if
    arbitrarily fine-grained.
    """
    all_units = data[entity_col].unique()
    if treated_unit not in all_units:
        raise ValueError(f"treated_unit {treated_unit!r} not found in data[{entity_col!r}]")

    if donor_units is None:
        donor_units = [u for u in all_units if u != treated_unit]
    else:
        missing = [u for u in donor_units if u not in all_units]
        if missing:
            raise ValueError(f"donor_units not found in data: {missing}")
        if treated_unit in donor_units:
            raise ValueError(f"treated_unit {treated_unit!r} cannot also be in donor_units")

    if len(donor_units) < 2:
        raise ValueError(
            f"synthetic_control_fit requires at least 2 donor units; got {len(donor_units)}"
        )

    result = _fit_single_unit(
        data, outcome, entity_col, period_col, treated_unit, treatment_period, donor_units
    )

    if run_placebo_inference:
        placebo_effects: dict[object, np.ndarray] = {}
        for placebo_unit in donor_units:
            placebo_donor_pool = [treated_unit] + [d for d in donor_units if d != placebo_unit]
            placebo_result = _fit_single_unit(
                data, outcome, entity_col, period_col, placebo_unit, treatment_period,
                placebo_donor_pool,
            )
            placebo_effects[placebo_unit] = placebo_result.effect_by_period

        all_avg_effects = {treated_unit: result.average_effect}
        all_avg_effects.update({u: float(eff.mean()) for u, eff in placebo_effects.items()})
        real_abs_effect = abs(result.average_effect)
        n_at_least_as_extreme = sum(
            1 for eff in all_avg_effects.values() if abs(eff) >= real_abs_effect
        )
        p_value = n_at_least_as_extreme / len(all_avg_effects)

        result.placebo_effects = placebo_effects
        result.placebo_p_value = p_value

    return result


def _fit_single_unit(
    data: pd.DataFrame,
    outcome: str,
    entity_col: str,
    period_col: str,
    target_unit: object,
    treatment_period: object,
    donor_units: list,
) -> SyntheticControlResult:
    """Core single-unit SCM fit, shared by both the real treated-unit fit
    and each placebo-unit fit in synthetic_control_fit's inference loop."""
    pivoted = data.pivot(index=entity_col, columns=period_col, values=outcome)
    periods = pivoted.columns.to_numpy()
    pre_mask = periods < treatment_period
    post_mask = ~pre_mask

    if not pre_mask.any():
        raise ValueError(
            f"treatment_period {treatment_period!r} leaves no pre-treatment periods "
            f"(all periods are >= treatment_period)."
        )
    if not post_mask.any():
        raise ValueError(
            f"treatment_period {treatment_period!r} leaves no post-treatment periods "
            f"(no periods are >= treatment_period)."
        )

    target_row = pivoted.loc[target_unit].to_numpy()
    donor_matrix = pivoted.loc[donor_units].to_numpy()

    target_pre = target_row[pre_mask]
    donor_pre = donor_matrix[:, pre_mask]

    weights_arr = _fit_weights(target_pre, donor_pre)
    weights = dict(zip(donor_units, weights_arr.tolist()))

    synthetic_full = weights_arr @ donor_matrix
    effect_full = target_row - synthetic_full
    effect_post = effect_full[post_mask]
    pre_rmse = float(np.sqrt(np.mean(effect_full[pre_mask] ** 2)))

    return SyntheticControlResult(
        treated_unit=target_unit,
        weights=weights,
        periods=periods,
        treatment_period=treatment_period,
        treated_outcome=target_row,
        synthetic_outcome=synthetic_full,
        effect_by_period=effect_post,
        average_effect=float(effect_post.mean()),
        pre_period_fit_rmse=pre_rmse,
    )
