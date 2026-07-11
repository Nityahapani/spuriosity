"""Tests for Confounder integration in spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.api as sm

from spuriosity import PanelGenerator


def test_confounder_induces_predicted_ovb_naive_regression():
    gen = PanelGenerator(n_entities=200_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.01)
    gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=False)

    df, truth = gen.generate()

    naive_slope = np.polyfit(df["x1"], df["y"], 1)[0]
    from spuriosity.pathologies import Confounder

    predicted_bias = Confounder(feature="x1", outcome="y", strength=0.6).predicted_naive_bias(
        feature_std=1.0
    )
    assert (naive_slope - 2.0) == pytest.approx(predicted_bias, abs=0.02)


def test_confounder_observed_false_hides_column():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_confounder(feature="x1", outcome="y", strength=0.5, observed=False)
    df, _ = gen.generate()
    assert "_confounder_x1" not in df.columns


def test_confounder_observed_true_exposes_column_and_recovers_true_coefficient():
    gen = PanelGenerator(n_entities=200_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.01)
    gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=True)

    df, _ = gen.generate()
    assert "_confounder_x1" in df.columns

    X = sm.add_constant(df[["x1", "_confounder_x1"]])
    model = sm.OLS(df["y"], X).fit()
    assert model.params["x1"] == pytest.approx(2.0, abs=0.02)


def test_confounder_ground_truth_records_strength():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_confounder(feature="x1", outcome="y", strength=0.6)
    _, truth = gen.generate()
    assert truth.confounding_strength == {"x1": 0.6}


def test_confounder_on_undeclared_feature_raises():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_confounder(feature="nonexistent", outcome="y", strength=0.5)
    with pytest.raises(ValueError, match="not a declared variable or treatment"):
        gen.generate()


def test_confounder_on_treatment_warns():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="treat", coefficients={"treat": 1.0})
    gen.add_confounder(feature="treat", outcome="y", strength=0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.generate()
        assert len(caught) == 1
        assert "treatment column" in str(caught[0].message)


def test_confounder_on_non_treatment_feature_does_not_warn():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 1.0})
    gen.add_confounder(feature="x1", outcome="y", strength=0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.generate()
        assert len(caught) == 0


def test_confounder_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=100, n_periods=5, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_confounder(feature="x1", outcome="y", strength=0.5, observed=True)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    import pandas.testing as pdt

    pdt.assert_frame_equal(df_a, df_b)


def test_confounder_and_structural_break_compose():
    """Sanity check that Confounder and StructuralBreak can be used
    together without interfering with each other's effects."""
    gen = PanelGenerator(n_entities=100_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_confounder(feature="x1", outcome="y", strength=0.4, observed=True)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=5.0)

    df, truth = gen.generate()
    assert "_confounder_x1" in df.columns
    assert len(truth.break_points) == 1
    assert truth.confounding_strength == {"x1": 0.4}

    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(5.0, abs=0.5)


def test_multiple_confounders_on_different_features():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_variable("x2", dist="normal", mean=0, std=1)
    gen.set_outcome(
        formula="x1 + x2", coefficients={"x1": 1.0, "x2": 1.0, "Intercept": 0.0}, noise_std=0.1
    )
    gen.add_confounder(feature="x1", outcome="y", strength=0.4)
    gen.add_confounder(feature="x2", outcome="y", strength=0.7)

    _, truth = gen.generate()
    assert truth.confounding_strength == {"x1": 0.4, "x2": 0.7}
