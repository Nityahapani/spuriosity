"""Tests for PanelGenerator.add_treatment(assignment="propensity", ...) --
covariate-dependent binary treatment assignment, closing a gap that
previously forced PSM/panel-FE-RE test scenarios to construct confounded
treatment by hand rather than through a first-class spuriosity API.
"""

from __future__ import annotations

import numpy as np
import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator


def test_propensity_assignment_correlates_with_covariate():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.8*x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    treated_mean_x1 = df[df["treat"] == 1]["x1"].mean()
    control_mean_x1 = df[df["treat"] == 0]["x1"].mean()
    assert treated_mean_x1 > 0.3
    assert control_mean_x1 < -0.3


def test_propensity_assignment_matches_sigmoid_formula_approximately():
    gen = PanelGenerator(n_entities=200_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.8*x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    # Bin by x1 and check the empirical treatment rate matches the
    # theoretical sigmoid(0.8*x1) reasonably closely within each bin.
    for x1_center in [-1.0, 0.0, 1.0]:
        window = df[(df["x1"] - x1_center).abs() < 0.1]
        empirical_rate = window["treat"].mean()
        predicted_rate = 1 / (1 + np.exp(-0.8 * x1_center))
        assert empirical_rate == pytest.approx(predicted_rate, abs=0.05)


def test_propensity_assignment_is_entity_fixed_across_periods():
    gen = PanelGenerator(n_entities=1000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.8*x1", start_period=0)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    per_entity_nunique = df.groupby("entity_id")["treat"].nunique()
    assert (per_entity_nunique == 1).all()


def test_propensity_assignment_respects_start_period():
    gen = PanelGenerator(n_entities=1000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.8*x1", start_period=3)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    pre = df[df["period"] < 3]
    assert (pre["treat"] == 0).all()


def test_propensity_assignment_requires_formula():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError, match="propensity_formula is required"):
        gen.add_treatment("treat", assignment="propensity")


def test_propensity_assignment_undeclared_variable_raises_at_generate_time():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.8*nonexistent")
    gen.set_outcome(formula="treat", coefficients={"treat": 1.0})
    with pytest.raises(ValueError, match="Failed to evaluate propensity_formula"):
        gen.generate()


def test_propensity_assignment_blocks_code_injection():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment(
        "treat", assignment="propensity", propensity_formula='__import__("os").system("echo pwned")'
    )
    gen.set_outcome(formula="treat", coefficients={"treat": 1.0})
    with pytest.raises(ValueError, match="Failed to evaluate propensity_formula"):
        gen.generate()


def test_invalid_assignment_value_rejected():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError, match="Unsupported assignment"):
        gen.add_treatment("treat", assignment="banana")


def test_propensity_assignment_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
        g.add_variable("x1")
        g.add_treatment("treat", assignment="propensity", propensity_formula="0.5*x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    pdt.assert_frame_equal(df_a, df_b)


def test_random_assignment_still_works_unchanged():
    """Backward-compatibility check: the pre-existing assignment='random'
    path must be completely unaffected by this extension."""
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=42)
    gen.add_treatment("treat", assignment="random", propensity=0.3)
    gen.set_outcome(formula="treat", coefficients={"treat": 1.0})
    df, truth = gen.generate()
    assert df["treat"].mean() == pytest.approx(0.3, abs=0.05)


def test_propensity_assignment_full_psm_scenario_via_first_class_api():
    """End-to-end integration: the full 'confounded binary treatment ->
    PSM recovers true ATE' scenario, now built entirely through
    PanelGenerator's own API rather than manual post-hoc construction."""
    from spuriosity import reference

    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.8*x1")
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 2.0, "treat": 3.0}, noise_std=0.5)
    df, truth = gen.generate()

    naive_effect = df[df.treat == 1]["y"].mean() - df[df.treat == 0]["y"].mean()
    assert abs(naive_effect - 3.0) > 0.3  # confirms confounding is real

    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.ate_estimate == pytest.approx(3.0, abs=0.1)
