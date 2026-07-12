"""Tests for spuriosity.reference — verified against known-true coefficients
from real PanelGenerator output, not synthetic ad-hoc arrays, so these
double as end-to-end checks of the generation -> fitting pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity import PanelGenerator, reference


def _basic_linear_data(n_entities: int = 50_000, seed: int = 42):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(
        formula="x1 + treat",
        coefficients={"x1": 2.0, "treat": 3.0, "Intercept": 1.0},
        noise_std=0.5,
    )
    return gen.generate()


# ----------------------------------------------------------------------
# OLS
# ----------------------------------------------------------------------


def test_ols_fit_recovers_true_coefficients():
    df, truth = _basic_linear_data()
    fit = reference.ols_fit(df, formula="y ~ x1 + treat")
    for key, true_val in truth.true_coefficients.items():
        assert fit.coefficients[key] == pytest.approx(true_val, abs=0.05)


def test_ols_predict_shape_and_type():
    df, _ = _basic_linear_data(n_entities=1000)
    fit = reference.ols_fit(df, formula="y ~ x1 + treat")
    preds = reference.ols_predict(fit, df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(df),)


def test_ols_predict_reasonably_close_to_actual_outcome():
    df, _ = _basic_linear_data(n_entities=10_000)
    fit = reference.ols_fit(df, formula="y ~ x1 + treat")
    preds = reference.ols_predict(fit, df)
    residual_std = (df["y"].to_numpy() - preds).std()
    assert residual_std < 1.0  # noise_std was 0.5, should be in this ballpark


# ----------------------------------------------------------------------
# sklearn LinearRegression
# ----------------------------------------------------------------------


def test_sklearn_lr_fit_recovers_true_coefficients():
    df, truth = _basic_linear_data()
    fit = reference.sklearn_lr_fit(df, features=["x1", "treat"], target="y")
    for key, true_val in truth.true_coefficients.items():
        assert fit.coefficients[key] == pytest.approx(true_val, abs=0.05)


def test_sklearn_lr_predict_matches_ols_predict_closely():
    """Both fit the same linear model via different libraries; predictions
    on the training data should closely agree."""
    df, _ = _basic_linear_data(n_entities=5000)
    ols_fit_result = reference.ols_fit(df, formula="y ~ x1 + treat")
    sklearn_fit_result = reference.sklearn_lr_fit(df, features=["x1", "treat"], target="y")

    ols_preds = reference.ols_predict(ols_fit_result, df)
    sklearn_preds = reference.sklearn_lr_predict(sklearn_fit_result, df)
    np.testing.assert_allclose(ols_preds, sklearn_preds, atol=0.01)


# ----------------------------------------------------------------------
# Difference-in-Differences
# ----------------------------------------------------------------------


def _did_data(n_entities: int = 5000, seed: int = 42, magnitude: float = 2.0):
    gen = PanelGenerator(n_entities=n_entities, n_periods=10, seed=seed)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(formula="treat", coefficients={"treat": 0.0, "Intercept": 1.0}, noise_std=0.1)
    gen.add_structural_break(
        period=5, target="y", kind="coefficient_shift", magnitude=magnitude, coefficient_target="treat"
    )
    return gen.generate()


def test_did_fit_recovers_true_effect_from_structural_break():
    df, _ = _did_data(magnitude=2.0)
    fit = reference.did_fit(df, outcome="y", treatment="treat", period="period", post_period=5)
    assert fit.ate_estimate == pytest.approx(2.0, abs=0.1)


def test_did_fit_ate_estimate_matches_interaction_coefficient():
    df, _ = _did_data()
    fit = reference.did_fit(df, outcome="y", treatment="treat", period="period", post_period=5)
    assert fit.ate_estimate == fit.coefficients["treat:post"]


def test_did_fit_does_not_mutate_input_dataframe():
    df, _ = _did_data(n_entities=100)
    assert "post" not in df.columns
    reference.did_fit(df, outcome="y", treatment="treat", period="period", post_period=5)
    assert "post" not in df.columns  # still unmutated after fit


def test_did_predict_requires_post_column():
    df, _ = _did_data(n_entities=100)
    fit = reference.did_fit(df, outcome="y", treatment="treat", period="period", post_period=5)
    with pytest.raises(ValueError, match="requires a 'post' column"):
        reference.did_predict(fit, df)


def test_did_predict_works_with_post_column_present():
    df, _ = _did_data(n_entities=1000)
    fit = reference.did_fit(df, outcome="y", treatment="treat", period="period", post_period=5)
    df_with_post = df.copy()
    df_with_post["post"] = (df_with_post["period"] >= 5).astype(int)
    preds = reference.did_predict(fit, df_with_post)
    assert preds.shape == (len(df),)


# ----------------------------------------------------------------------
# Logit
# ----------------------------------------------------------------------


def _bernoulli_logit_data(n_entities: int = 20_000, seed: int = 1, true_slope: float = 0.5):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)

    def outcome_fn(x1):
        p = 1 / (1 + np.exp(-(true_slope * x1)))
        rng = np.random.default_rng(123)
        return (rng.random(len(x1)) < p).astype(float)

    gen.set_outcome(fn=outcome_fn, noise_std=0.0)
    return gen.generate()


def test_logit_fit_recovers_true_slope():
    df, _ = _bernoulli_logit_data(true_slope=0.5)
    fit = reference.logit_fit(df, formula="y ~ x1")
    assert fit.coefficients["x1"] == pytest.approx(0.5, abs=0.1)
    assert fit.coefficients["Intercept"] == pytest.approx(0.0, abs=0.1)


def test_logit_predict_returns_probabilities_in_unit_interval():
    df, _ = _bernoulli_logit_data(n_entities=1000)
    fit = reference.logit_fit(df, formula="y ~ x1")
    preds = reference.logit_predict(fit, df)
    assert (preds >= 0).all()
    assert (preds <= 1).all()


# ----------------------------------------------------------------------
# DoubleML (optional dependency)
# ----------------------------------------------------------------------


def _confounded_treatment_data(n_entities: int = 3000, seed: int = 42):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 3.0}, noise_std=0.5)
    gen.add_confounder(feature="x1", outcome="y", strength=0.4, observed=True)
    return gen.generate()


def test_doubleml_fit_recovers_true_ate():
    doubleml = pytest.importorskip("doubleml")
    del doubleml
    df, truth = _confounded_treatment_data()
    fit = reference.doubleml_fit(df, outcome="y", treatment="treat", covariates=["x1", "_confounder_x1"])
    assert fit.ate_estimate == pytest.approx(truth.treatment_effect_ate, abs=0.3)


def test_doubleml_predict_broadcasts_ate_estimate():
    doubleml = pytest.importorskip("doubleml")
    del doubleml
    df, _ = _confounded_treatment_data(n_entities=500)
    fit = reference.doubleml_fit(df, outcome="y", treatment="treat", covariates=["x1", "_confounder_x1"])
    preds = reference.doubleml_predict(fit, df)
    assert preds.shape == (len(df),)
    assert (preds == fit.ate_estimate).all()


def test_doubleml_predict_without_ate_estimate_raises():
    import pandas as pd

    from spuriosity.reference import FitResult

    fit = FitResult(coefficients={}, raw_model=None, ate_estimate=None)
    with pytest.raises(ValueError, match="no ate_estimate"):
        reference.doubleml_predict(fit, pd.DataFrame({"x": [1, 2]}))
