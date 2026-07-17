"""Tests for spuriosity.reference.xgboost_fit / xgboost_predict --
a pure predictive ML baseline for the "does a flexible nonlinear model
beat econometric assumptions" comparison."""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity import PanelGenerator, compare_models, reference


def _nonlinear_data(n_entities: int = 10_000, seed: int = 42):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_variable("x2", dist="normal", mean=0, std=1)
    gen.set_outcome(
        fn=lambda x1, x2: 2 * x1 + x1**2 - 0.5 * x2 + np.sin(x2 * 2), noise_std=0.3
    )
    return gen.generate()


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ----------------------------------------------------------------------
# Core fit/predict
# ----------------------------------------------------------------------


def test_xgboost_fit_returns_empty_coefficients():
    df, truth = _nonlinear_data(n_entities=500)
    fit = reference.xgboost_fit(df, outcome="y", features=["x1", "x2"])
    assert fit.coefficients == {}


def test_xgboost_predict_shape():
    df, truth = _nonlinear_data(n_entities=500)
    fit = reference.xgboost_fit(df, outcome="y", features=["x1", "x2"])
    preds = reference.xgboost_predict(fit, df)
    assert preds.shape == (len(df),)


def test_xgboost_feature_importances_present_and_sum_reasonably():
    df, truth = _nonlinear_data(n_entities=2000)
    fit = reference.xgboost_fit(df, outcome="y", features=["x1", "x2"])
    importances = fit.extra["feature_importances"]
    assert set(importances.keys()) == {"x1", "x2"}
    assert all(v >= 0 for v in importances.values())
    assert sum(importances.values()) == pytest.approx(1.0, abs=0.01)


def test_xgboost_outperforms_ols_on_nonlinear_data():
    df, truth = _nonlinear_data(n_entities=10_000)

    fit_xgb = reference.xgboost_fit(df, outcome="y", features=["x1", "x2"])
    preds_xgb = reference.xgboost_predict(fit_xgb, df)
    r2_xgb = _r_squared(df["y"].to_numpy(), preds_xgb)

    fit_ols = reference.ols_fit(df, formula="y ~ x1 + x2")
    preds_ols = reference.ols_predict(fit_ols, df)
    r2_ols = _r_squared(df["y"].to_numpy(), preds_ols)

    assert r2_xgb > r2_ols + 0.1


def test_xgboost_does_not_meaningfully_beat_ols_on_linear_data():
    """Sanity check the other direction: on genuinely linear data, a
    flexible nonlinear model shouldn't have a large predictive advantage
    over correctly-specified OLS (and may do slightly worse due to
    overfitting on a modest sample) -- confirms the earlier win on
    nonlinear data reflects real functional-form flexibility, not just
    XGBoost being unconditionally 'better'."""
    gen = PanelGenerator(n_entities=5000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=1.0)
    df, truth = gen.generate()

    fit_xgb = reference.xgboost_fit(df, outcome="y", features=["x1"])
    preds_xgb = reference.xgboost_predict(fit_xgb, df)
    r2_xgb = _r_squared(df["y"].to_numpy(), preds_xgb)

    fit_ols = reference.ols_fit(df, formula="y ~ x1")
    preds_ols = reference.ols_predict(fit_ols, df)
    r2_ols = _r_squared(df["y"].to_numpy(), preds_ols)

    assert r2_xgb <= r2_ols + 0.05  # no meaningful advantage on linear-truth data


def test_xgboost_accepts_extra_kwargs():
    df, truth = _nonlinear_data(n_entities=500)
    fit = reference.xgboost_fit(df, outcome="y", features=["x1", "x2"], learning_rate=0.05)
    assert fit.raw_model.get_params()["learning_rate"] == 0.05


def test_xgboost_custom_n_estimators_and_max_depth():
    df, truth = _nonlinear_data(n_entities=500)
    fit = reference.xgboost_fit(
        df, outcome="y", features=["x1", "x2"], n_estimators=50, max_depth=2
    )
    assert fit.raw_model.get_params()["n_estimators"] == 50
    assert fit.raw_model.get_params()["max_depth"] == 2


# ----------------------------------------------------------------------
# compare_models integration
# ----------------------------------------------------------------------


def test_xgboost_excluded_from_coef_rmse_ranking_but_ols_included():
    gen = PanelGenerator(n_entities=5000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.5)
    df, truth = gen.generate()

    results = compare_models(
        data=df,
        truth=truth,
        models={
            "OLS": (reference.ols_fit, reference.ols_predict),
            "XGBoost": (reference.xgboost_fit, reference.xgboost_predict),
        },
        fit_kwargs_per_model={
            "OLS": {"formula": "y ~ x1"},
            "XGBoost": {"outcome": "y", "features": ["x1"]},
        },
    )
    table = results.ranked_table(by="coef_rmse")
    assert list(table["model"]) == ["OLS"]
    assert table.attrs["excluded_models"] == ["XGBoost"]


def test_xgboost_and_ols_both_appear_in_reports_regardless_of_ranking():
    gen = PanelGenerator(n_entities=5000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.5)
    df, truth = gen.generate()

    results = compare_models(
        data=df,
        truth=truth,
        models={
            "OLS": (reference.ols_fit, reference.ols_predict),
            "XGBoost": (reference.xgboost_fit, reference.xgboost_predict),
        },
        fit_kwargs_per_model={
            "OLS": {"formula": "y ~ x1"},
            "XGBoost": {"outcome": "y", "features": ["x1"]},
        },
    )
    assert set(results.reports.keys()) == {"OLS", "XGBoost"}
