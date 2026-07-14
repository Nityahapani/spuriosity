"""Tests for MeasurementError integration in
spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator, reference


def test_measurement_error_produces_attenuation_bias():
    gen = PanelGenerator(n_entities=1_000_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.01)
    gen.add_measurement_error(feature="x1", noise_std=0.8)
    df, truth = gen.generate()

    predicted_beta = 2.0 * truth.measurement_error[0].reliability_ratio
    fit = reference.ols_fit(df, formula="y ~ x1")
    assert fit.coefficients["x1"] == pytest.approx(predicted_beta, abs=0.02)


def test_outcome_is_built_from_true_value_not_observed_value():
    """The outcome must reflect the TRUE pre-noise feature value, not the
    corrupted observed column -- verified by checking that residuals
    computed using the observed (noisy) feature are large, reflecting the
    measurement noise itself rather than the outcome's own small noise."""
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.01)
    gen.add_measurement_error(feature="x1", noise_std=0.8)
    df, truth = gen.generate()

    resid_using_observed = df["y"] - 2.0 * df["x1"]
    assert resid_using_observed.std() > 0.5


def test_measurement_error_ground_truth_recorded():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_measurement_error(feature="x1", noise_std=0.5)
    _, truth = gen.generate()
    assert len(truth.measurement_error) == 1
    assert truth.measurement_error[0].feature == "x1"
    assert truth.measurement_error[0].noise_std == 0.5
    assert 0.0 < truth.measurement_error[0].reliability_ratio < 1.0


def test_measurement_error_absent_when_not_used():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.measurement_error == []


def test_measurement_error_on_undeclared_feature_raises():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_measurement_error(feature="nonexistent", noise_std=0.5)
    with pytest.raises(ValueError, match="not a declared variable or treatment"):
        gen.generate()


def test_measurement_error_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_measurement_error(feature="x1", noise_std=0.5)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    pdt.assert_frame_equal(df_a, df_b)


def test_measurement_error_zero_noise_leaves_data_unchanged():
    gen_with = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
    gen_with.add_variable("x1")
    gen_with.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
    gen_with.add_measurement_error(feature="x1", noise_std=0.0)
    df_with, _ = gen_with.generate()

    gen_without = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
    gen_without.add_variable("x1")
    gen_without.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
    df_without, _ = gen_without.generate()

    pdt.assert_frame_equal(df_with, df_without)


def test_measurement_error_composes_with_structural_break():
    gen = PanelGenerator(n_entities=200_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_measurement_error(feature="x1", noise_std=0.5)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=3.0)
    df, truth = gen.generate()

    assert len(truth.measurement_error) == 1
    assert len(truth.break_points) == 1
    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(3.0, abs=0.3)


def test_measurement_error_composes_with_confounder_on_same_feature():
    """Confounder modifies the feature BEFORE the outcome is computed
    (affecting the true relationship); MeasurementError corrupts it AFTER
    (affecting only what's observed). Both should apply without error when
    targeting the same feature."""
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_confounder(feature="x1", outcome="y", strength=0.3, observed=True)
    gen.add_measurement_error(feature="x1", noise_std=0.5)
    df, truth = gen.generate()

    assert "_confounder_x1" in df.columns
    assert truth.confounding_strength == {"x1": 0.3}
    assert len(truth.measurement_error) == 1
