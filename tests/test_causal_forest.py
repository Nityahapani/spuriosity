"""Tests for spuriosity.reference.causal_forest_fit / causal_forest_predict.

Verified against real spuriosity-generated HTE data (via add_hte), which
provides a genuine known true_cate function to check recovery against --
these tests double as integration tests of add_hte -> causal_forest_fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity import PanelGenerator, reference


def _hte_data(n_entities: int = 5000, seed: int = 42, formula: str = "3 + 1.5*x1"):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + treat", coefficients={"x1": 2.0, "treat": 0.0, "Intercept": 0.0},
        noise_std=0.1,
    )
    gen.add_hte(treatment="treat", modifier="x1", formula=formula)
    return gen.generate()


# ----------------------------------------------------------------------
# Core fit/predict
# ----------------------------------------------------------------------


def test_causal_forest_fit_returns_empty_coefficients():
    df, truth = _hte_data(n_entities=500)
    fit = reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.coefficients == {}


def test_causal_forest_predict_returns_per_row_cate_not_outcome():
    df, truth = _hte_data(n_entities=500)
    fit = reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    preds = reference.causal_forest_predict(fit, df)
    assert preds.shape == (len(df),)
    # sanity: predictions should look like small treatment-effect values
    # (roughly 0-10 for this DGP), not outcome-scale values
    assert preds.mean() == pytest.approx(3.0, abs=1.0)


def test_causal_forest_ate_estimate_matches_average_cate():
    df, truth = _hte_data(n_entities=5000)
    fit = reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    # E[x1] = 0 for standard normal -> E[3 + 1.5*x1] = 3
    assert fit.ate_estimate == pytest.approx(3.0, abs=0.3)
    assert fit.extra["ate_estimate"] == fit.ate_estimate


def test_causal_forest_recovers_true_cate_with_high_correlation():
    df, truth = _hte_data(n_entities=5000)
    fit = reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    per_row_cate = reference.causal_forest_predict(fit, df)
    true_cate_per_row = df["x1"].apply(truth.true_cate).to_numpy()

    corr = np.corrcoef(per_row_cate, true_cate_per_row)[0, 1]
    assert corr > 0.8


def test_causal_forest_beats_naive_linear_interaction_on_nonlinear_cate():
    """The real value-add case: when the true CATE is nonlinear (here
    quadratic), a causal forest should recover it more accurately than a
    naive OLS-with-linear-interaction specification, which is
    misspecified for a quadratic effect."""
    df, truth = _hte_data(n_entities=10_000, formula="3 + 1.5*x1 + 0.5*x1**2")

    fit_ols = reference.ols_fit(df, formula="y ~ x1*treat")
    naive_cate_at_2 = fit_ols.coefficients.get("x1:treat", 0.0) * 2.0 + fit_ols.coefficients.get(
        "treat", 0.0
    )

    fit_cf = reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    cf_cate_at_2 = fit_cf.raw_model.effect(np.array([[2.0]]))[0]

    true_cate_at_2 = truth.true_cate(2.0)
    naive_error = abs(naive_cate_at_2 - true_cate_at_2)
    cf_error = abs(cf_cate_at_2 - true_cate_at_2)

    assert cf_error < naive_error


# ----------------------------------------------------------------------
# n_estimators / subforest_size validation
# ----------------------------------------------------------------------


def test_n_estimators_not_divisible_by_four_raises_clear_error():
    df, truth = _hte_data(n_entities=200)
    with pytest.raises(ValueError, match="must be evenly divisible"):
        reference.causal_forest_fit(
            df, outcome="y", treatment="treat", covariates=["x1"], n_estimators=50
        )


def test_n_estimators_divisible_by_four_works():
    df, truth = _hte_data(n_entities=200)
    fit = reference.causal_forest_fit(
        df, outcome="y", treatment="treat", covariates=["x1"], n_estimators=52
    )
    assert fit.ate_estimate is not None


def test_default_n_estimators_is_valid():
    """The default n_estimators=200 must itself satisfy the divisibility
    constraint -- a regression guard against changing the default to
    something invalid."""
    df, truth = _hte_data(n_entities=200)
    fit = reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.ate_estimate is not None


# ----------------------------------------------------------------------
# Warning-free classifier specification
# ----------------------------------------------------------------------


def test_no_classifier_mismatch_warning_with_default_discrete_treatment():
    import warnings

    df, truth = _hte_data(n_entities=500)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])
        classifier_warnings = [w for w in caught if "classifier" in str(w.message).lower()]
        assert len(classifier_warnings) == 0


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------


def test_causal_forest_reproducible_with_same_random_state():
    df, truth = _hte_data(n_entities=1000)
    fit_a = reference.causal_forest_fit(
        df, outcome="y", treatment="treat", covariates=["x1"], random_state=42, n_estimators=52
    )
    fit_b = reference.causal_forest_fit(
        df, outcome="y", treatment="treat", covariates=["x1"], random_state=42, n_estimators=52
    )
    preds_a = reference.causal_forest_predict(fit_a, df)
    preds_b = reference.causal_forest_predict(fit_b, df)
    np.testing.assert_allclose(preds_a, preds_b)


# ----------------------------------------------------------------------
# compare_models integration
# ----------------------------------------------------------------------


def test_causal_forest_excluded_from_coef_rmse_but_present_in_reports():
    from spuriosity import compare_models

    df, truth = _hte_data(n_entities=1000)
    results = compare_models(
        data=df,
        truth=truth,
        models={
            "OLS": (reference.ols_fit, reference.ols_predict),
            "CausalForest": (reference.causal_forest_fit, reference.causal_forest_predict),
        },
        fit_kwargs_per_model={
            "OLS": {"formula": "y ~ x1 + treat"},
            "CausalForest": {"outcome": "y", "treatment": "treat", "covariates": ["x1"]},
        },
    )
    assert set(results.reports.keys()) == {"OLS", "CausalForest"}
    table = results.ranked_table(by="coef_rmse")
    assert "CausalForest" not in list(table["model"])
