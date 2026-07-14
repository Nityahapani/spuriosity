"""Tests for Heteroskedasticity and Multicollinearity integration in
spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import pandas.testing as pdt
import pytest
import statsmodels.api as sm

from spuriosity import PanelGenerator, reference


# ----------------------------------------------------------------------
# Heteroskedasticity
# ----------------------------------------------------------------------


def test_heteroskedasticity_unbiased_point_estimate():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=1.0)
    gen.add_heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
    df, truth = gen.generate()

    fit = reference.ols_fit(df, formula="y ~ x1")
    assert fit.coefficients["x1"] == pytest.approx(2.0, abs=0.05)


def test_heteroskedasticity_robust_se_exceeds_naive_se():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=1.0)
    gen.add_heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
    df, truth = gen.generate()

    X = sm.add_constant(df["x1"])
    model_naive = sm.OLS(df["y"], X).fit()
    model_robust = sm.OLS(df["y"], X).fit(cov_type="HC3")
    assert model_robust.bse["x1"] > model_naive.bse["x1"] * 1.3


def test_heteroskedasticity_ground_truth_recorded():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
    _, truth = gen.generate()
    assert len(truth.heteroskedasticity) == 1
    assert truth.heteroskedasticity[0].feature == "x1"
    assert truth.heteroskedasticity[0].formula == "1 + 0.5*x1**2"


def test_heteroskedasticity_absent_when_not_used():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.heteroskedasticity == []


def test_heteroskedasticity_on_undeclared_feature_raises():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_heteroskedasticity(feature="nonexistent", formula="1 + x1")
    with pytest.raises(ValueError, match="not a declared variable or treatment"):
        gen.generate()


def test_heteroskedasticity_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    pdt.assert_frame_equal(df_a, df_b)


def test_multiple_heteroskedasticity_compose_multiplicatively():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=1.0)
    gen.add_heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
    gen.add_heteroskedasticity(feature="x1", formula="2.0")  # flat doubling on top
    df, truth = gen.generate()
    assert len(truth.heteroskedasticity) == 2

    # near x1=0: base multiplier ~1, second multiplier 2.0 -> effective std ~2.0
    near_zero = df[df["x1"].abs() < 0.1]
    resid_near_zero = near_zero["y"] - near_zero["x1"]
    assert resid_near_zero.std() == pytest.approx(2.0, abs=0.3)


# ----------------------------------------------------------------------
# Multicollinearity
# ----------------------------------------------------------------------


def test_multicollinearity_achieves_target_correlation():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    df, truth = gen.generate()

    actual_corr = df["x1"].corr(df["x2"])
    assert actual_corr == pytest.approx(0.9, abs=0.01)


def test_multicollinearity_creates_new_column():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    df, _ = gen.generate()
    assert "x2" in df.columns


def test_multicollinearity_ground_truth_recorded():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    _, truth = gen.generate()
    assert len(truth.multicollinearity) == 1
    assert truth.multicollinearity[0].feature == "x2"
    assert truth.multicollinearity[0].correlated_with == "x1"
    assert truth.multicollinearity[0].target_correlation == 0.9


def test_multicollinearity_feature_name_collision_at_builder_time():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_variable("x2")
    with pytest.raises(ValueError, match="already in use"):
        gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)


def test_add_variable_after_multicollinearity_with_same_name_raises():
    """Reverse-order collision: declaring a variable whose name was
    already claimed by an earlier add_multicollinearity() call."""
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    with pytest.raises(ValueError, match="already in use"):
        gen.add_variable("x2")


def test_multicollinearity_undeclared_correlated_with_raises_at_generate_time():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    with pytest.raises(ValueError, match="not a declared variable or treatment"):
        gen.generate()


def test_multicollinearity_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.8)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    pdt.assert_frame_equal(df_a, df_b)


def test_multicollinearity_absent_when_not_used():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.multicollinearity == []


# ----------------------------------------------------------------------
# Composition with other pathologies
# ----------------------------------------------------------------------


def test_heteroskedasticity_and_multicollinearity_compose_with_structural_break():
    gen = PanelGenerator(n_entities=100_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.5)
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.85)
    gen.add_heteroskedasticity(feature="x1", formula="1 + 0.3*x1**2")
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=3.0)

    df, truth = gen.generate()
    assert "x2" in df.columns
    assert len(truth.multicollinearity) == 1
    assert len(truth.heteroskedasticity) == 1
    assert len(truth.break_points) == 1

    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(3.0, abs=0.3)
