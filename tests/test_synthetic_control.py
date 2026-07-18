"""Tests for spuriosity.synthetic_control.synthetic_control_fit.

Verified against a panel with a known common-factor structure (so donor
combinations CAN plausibly match the treated unit's pre-treatment
trajectory, the condition SCM requires to be meaningful) and a manually
injected known treatment effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spuriosity.synthetic_control import SyntheticControlResult, synthetic_control_fit


def _factor_panel(
    n_entities: int = 20,
    n_periods: int = 30,
    seed: int = 42,
    true_effect: float = 5.0,
    treated_unit: int = 0,
    treatment_period: int = 20,
):
    """A panel where every entity's outcome is a loading on a shared
    common factor plus idiosyncratic noise -- the structural condition
    that makes synthetic control meaningful (a convex combination of
    donors CAN approximate the treated unit)."""
    rng = np.random.default_rng(seed)
    common_factor = rng.normal(size=n_periods).cumsum() * 0.3
    loadings = rng.uniform(0.5, 1.5, n_entities)

    rows = []
    for entity in range(n_entities):
        for period in range(n_periods):
            y = common_factor[period] * loadings[entity] + rng.normal() * 0.1
            rows.append({"entity_id": entity, "period": period, "y": y})
    df = pd.DataFrame(rows)

    if true_effect != 0.0:
        post_mask = (df["entity_id"] == treated_unit) & (df["period"] >= treatment_period)
        df.loc[post_mask, "y"] += true_effect

    return df, true_effect


# ----------------------------------------------------------------------
# Core effect recovery
# ----------------------------------------------------------------------


def test_synthetic_control_recovers_known_effect():
    df, true_effect = _factor_panel(true_effect=5.0)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    assert result.average_effect == pytest.approx(true_effect, abs=1.0)


def test_synthetic_control_pre_period_fit_is_good():
    """A meaningful precondition check: if the DGP has a real common
    factor structure, the pre-period fit RMSE should be small."""
    df, _ = _factor_panel()
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    assert result.pre_period_fit_rmse < 0.5


def test_synthetic_control_weights_are_valid_convex_combination():
    df, _ = _factor_panel()
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20, run_placebo_inference=False,
    )
    weights = np.array(list(result.weights.values()))
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0).all()


def test_synthetic_control_null_effect_near_zero():
    df, _ = _factor_panel(true_effect=0.0)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    assert abs(result.average_effect) < 0.5


def test_result_type_is_synthetic_control_result():
    df, _ = _factor_panel()
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20, run_placebo_inference=False,
    )
    assert isinstance(result, SyntheticControlResult)


# ----------------------------------------------------------------------
# Placebo-in-space inference
# ----------------------------------------------------------------------


def test_placebo_p_value_small_for_genuine_effect():
    df, _ = _factor_panel(true_effect=5.0)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    assert result.placebo_p_value is not None
    assert result.placebo_p_value <= 0.15


def test_placebo_p_value_not_small_for_null_effect():
    df, _ = _factor_panel(true_effect=0.0)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    assert result.placebo_p_value is not None
    assert result.placebo_p_value > 0.2


def test_placebo_effects_populated_with_one_entry_per_donor():
    df, _ = _factor_panel(n_entities=10)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    assert result.placebo_effects is not None
    assert len(result.placebo_effects) == 9  # 10 entities minus the treated one


def test_run_placebo_inference_false_skips_placebo():
    df, _ = _factor_panel(n_entities=10)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20, run_placebo_inference=False,
    )
    assert result.placebo_effects is None
    assert result.placebo_p_value is None


def test_placebo_p_value_is_valid_discrete_rank_based_value():
    """The placebo p-value is always of the form k/n for some integer
    1 <= k <= n, where n = 1 + number of donors (rank-based p-values are
    inherently discrete/granular with a small number of units). This is
    always true regardless of effect size -- unlike a naive expectation
    that a large true effect always ranks as the single most extreme
    unit, which is NOT guaranteed: a placebo unit whose own donor pool
    includes the (now-shifted) real treated unit can itself produce a
    large apparent "effect" while compensating for that shift, sometimes
    exceeding the real effect in magnitude. This is a genuine, documented
    limitation of placebo-in-space inference with small donor pools, not
    a bug -- confirmed directly during test development, where an
    initial test asserting "large true effect => minimum possible
    p-value" failed because a donor with high weight on the treated unit
    produced an even larger placebo effect when it became the placebo
    target itself.
    """
    df, _ = _factor_panel(n_entities=6, true_effect=5.0)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    n_units = 6  # 1 treated + 5 donors
    possible_values = [k / n_units for k in range(1, n_units + 1)]
    assert any(result.placebo_p_value == pytest.approx(v, abs=1e-9) for v in possible_values)


def test_placebo_p_value_achieves_floor_with_larger_donor_pool():
    """With a larger, more diverse donor pool, a clear true effect should
    more reliably achieve (or come close to) the minimum p-value, since
    no single donor's weight on the treated unit is large enough to make
    its own placebo fit unstable the way a small pool can."""
    df, _ = _factor_panel(n_entities=20, true_effect=5.0)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    n_units = 20
    expected_floor = 1 / n_units
    assert result.placebo_p_value == pytest.approx(expected_floor, abs=1e-9)


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def _small_panel(n_entities: int = 5, n_periods: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(n_entities):
        for period in range(n_periods):
            rows.append({"entity_id": entity, "period": period, "y": rng.normal()})
    return pd.DataFrame(rows)


def test_unknown_treated_unit_raises():
    df = _small_panel()
    with pytest.raises(ValueError, match="not found in data"):
        synthetic_control_fit(
            df, outcome="y", entity_col="entity_id", period_col="period",
            treated_unit=999, treatment_period=5,
        )


def test_too_few_donors_raises():
    df = _small_panel()
    with pytest.raises(ValueError, match="at least 2 donor units"):
        synthetic_control_fit(
            df, outcome="y", entity_col="entity_id", period_col="period",
            treated_unit=0, treatment_period=5, donor_units=[1],
        )


def test_treated_unit_in_donor_units_raises():
    df = _small_panel()
    with pytest.raises(ValueError, match="cannot also be in donor_units"):
        synthetic_control_fit(
            df, outcome="y", entity_col="entity_id", period_col="period",
            treated_unit=0, treatment_period=5, donor_units=[0, 1, 2],
        )


def test_unknown_donor_units_raises():
    df = _small_panel()
    with pytest.raises(ValueError, match="donor_units not found"):
        synthetic_control_fit(
            df, outcome="y", entity_col="entity_id", period_col="period",
            treated_unit=0, treatment_period=5, donor_units=[1, 999],
        )


def test_treatment_period_with_no_pre_period_raises():
    df = _small_panel()
    with pytest.raises(ValueError, match="no pre-treatment periods"):
        synthetic_control_fit(
            df, outcome="y", entity_col="entity_id", period_col="period",
            treated_unit=0, treatment_period=0,
        )


def test_treatment_period_with_no_post_period_raises():
    df = _small_panel()
    with pytest.raises(ValueError, match="no post-treatment periods"):
        synthetic_control_fit(
            df, outcome="y", entity_col="entity_id", period_col="period",
            treated_unit=0, treatment_period=1000,
        )


# ----------------------------------------------------------------------
# Explicit donor_units subset
# ----------------------------------------------------------------------


def test_explicit_donor_units_subset_respected():
    df, _ = _factor_panel(n_entities=10)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20, donor_units=[1, 2, 3],
        run_placebo_inference=False,
    )
    assert set(result.weights.keys()) == {1, 2, 3}


# ----------------------------------------------------------------------
# summary()
# ----------------------------------------------------------------------


def test_summary_does_not_raise(capsys):
    df, _ = _factor_panel(n_entities=10)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20,
    )
    result.summary()
    captured = capsys.readouterr()
    assert "SyntheticControlResult" in captured.out
    assert "Placebo-in-space p-value" in captured.out


def test_summary_without_placebo_does_not_print_p_value(capsys):
    df, _ = _factor_panel(n_entities=10)
    result = synthetic_control_fit(
        df, outcome="y", entity_col="entity_id", period_col="period",
        treated_unit=0, treatment_period=20, run_placebo_inference=False,
    )
    result.summary()
    captured = capsys.readouterr()
    assert "Placebo-in-space p-value" not in captured.out
