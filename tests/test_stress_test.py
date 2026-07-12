"""Tests for spuriosity.stress_test.StressTest."""

from __future__ import annotations

import pytest

from spuriosity import PanelGenerator, reference
from spuriosity.stress_test import StressTest, StressTestReport


def test_coef_rmse_near_zero_for_accurate_model():
    gen = PanelGenerator(n_entities=50_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(
        formula="x1 + treat", coefficients={"x1": 2.0, "treat": 3.0, "Intercept": 1.0}, noise_std=0.5
    )
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit,
        predict_fn=reference.ols_predict,
        data=df,
        fit_kwargs={"formula": "y ~ x1 + treat"},
        model_name="OLS",
    )
    assert report.metrics["coef_rmse"] < 0.05


def test_report_is_stress_test_report_instance():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"},
    )
    assert isinstance(report, StressTestReport)


def test_coef_rmse_absent_when_no_shared_keys():
    """If the fitted model's coefficients share no keys with the true
    coefficients, coef_rmse should not silently report 0 or an error."""
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    def weird_fit(data):
        return {"totally_unrelated_key": 42.0}

    test = StressTest(truth)
    report = test.evaluate(fit_fn=weird_fit, predict_fn=lambda f, d: None, data=df, fit_kwargs={})
    assert "coef_rmse" not in report.metrics


def test_confounding_bias_matches_closed_form_prediction():
    gen = PanelGenerator(n_entities=200_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.01)
    gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=False)
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"}, model_name="naive_OLS",
    )
    predicted_bias = 0.6**2 / (1.0**2 + 0.6**2)
    assert report.metrics["confounding_bias"] == pytest.approx(predicted_bias, abs=0.02)
    assert report.metrics["confounding_bias:x1"] == report.metrics["confounding_bias"]


def test_confounding_bias_near_zero_when_confounder_controlled_for():
    gen = PanelGenerator(n_entities=200_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.01)
    gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=True)
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1 + _confounder_x1"}, model_name="controlled_OLS",
    )
    assert report.metrics["confounding_bias"] < 0.02


def test_confounding_bias_absent_without_confounder():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"},
    )
    assert "confounding_bias" not in report.metrics


def test_break_detection_lag_zero_for_clean_break():
    gen = PanelGenerator(n_entities=2000, n_periods=20, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.3)
    gen.add_structural_break(
        period=10, target="y", kind="coefficient_shift", magnitude=4.0, coefficient_target="x1"
    )
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"}, model_name="OLS",
    )
    assert report.metrics["break_detection_lag"] == pytest.approx(0.0, abs=1.0)


def test_break_detection_lag_absent_without_break():
    gen = PanelGenerator(n_entities=1000, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"},
    )
    assert "break_detection_lag" not in report.metrics


def test_break_detection_lag_absent_without_formula_kwarg():
    """The break-detection-lag rolling refit needs a formula= kwarg to
    reuse across windows; without one it should be skipped, not error."""
    gen = PanelGenerator(n_entities=1000, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_structural_break(
        period=5, target="y", kind="coefficient_shift", magnitude=2.0, coefficient_target="x1"
    )
    df, truth = gen.generate()

    def fit_no_formula_kwarg(data, features):
        return reference.sklearn_lr_fit(data, features=features, target="y")

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=fit_no_formula_kwarg,
        predict_fn=lambda f, d: None,
        data=df,
        fit_kwargs={"features": ["x1"]},
    )
    assert "break_detection_lag" not in report.metrics


def test_summary_does_not_raise(capsys):
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"}, model_name="OLS",
    )
    report.summary()
    captured = capsys.readouterr()
    assert "OLS" in captured.out


def test_model_name_recorded_in_report():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"}, model_name="my_custom_model_name",
    )
    assert report.model_name == "my_custom_model_name"
