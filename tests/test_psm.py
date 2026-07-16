"""Tests for spuriosity.reference.psm_fit / psm_predict.

The PSM test scenario (a confounded binary treatment where propensity
depends on an observed covariate) is constructed by hand on top of
spuriosity-generated data, since spuriosity does not yet have a dedicated
pathology for generating a covariate-dependent binary treatment
assignment (add_treatment's "random" assignment is independent of
covariates by design) -- a natural candidate for a future pathology.
"""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity import PanelGenerator, reference
from spuriosity.reference import FitResult


def _confounded_treatment_data(n: int = 100_000, seed: int = 42, propensity_strength: float = 0.8):
    rng = np.random.default_rng(seed)
    gen = PanelGenerator(n_entities=n, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(fn=lambda x1: 2.0 * x1, noise_std=0.5)
    df, truth = gen.generate()

    propensity_true = 1 / (1 + np.exp(-(propensity_strength * df["x1"].to_numpy())))
    treat = (rng.random(n) < propensity_true).astype(int)
    df = df.copy()
    df["treat"] = treat
    true_ate = 3.0
    df["y"] = df["y"] + true_ate * df["treat"]
    return df, true_ate


# ----------------------------------------------------------------------
# Core recovery
# ----------------------------------------------------------------------


def test_psm_recovers_true_ate():
    df, true_ate = _confounded_treatment_data()
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.ate_estimate == pytest.approx(true_ate, abs=0.1)


def test_naive_diff_in_means_is_biased_on_same_data():
    """Confirms the test scenario actually produces confounding bias --
    i.e. that PSM's correction is doing real work, not just matching a
    already-unbiased naive estimate."""
    df, true_ate = _confounded_treatment_data()
    naive_effect = df[df.treat == 1]["y"].mean() - df[df.treat == 0]["y"].mean()
    assert abs(naive_effect - true_ate) > 0.3


def test_psm_ate_estimate_matches_coefficients_dict():
    df, true_ate = _confounded_treatment_data(n=5000)
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.coefficients["treat"] == fit.ate_estimate


def test_psm_ate_std_error_is_positive():
    df, true_ate = _confounded_treatment_data(n=5000)
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.ate_std_error is not None
    assert fit.ate_std_error > 0


# ----------------------------------------------------------------------
# Common support diagnostic
# ----------------------------------------------------------------------


def test_common_support_high_for_good_overlap():
    df, _ = _confounded_treatment_data(propensity_strength=0.8)
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.extra["common_support_fraction"] > 0.99


def test_common_support_lower_for_poor_overlap():
    df, _ = _confounded_treatment_data(n=20_000, propensity_strength=4.0)  # extreme separation
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert fit.extra["common_support_fraction"] < 0.99


def test_propensity_scores_stored_and_correct_length():
    df, _ = _confounded_treatment_data(n=5000)
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    assert len(fit.extra["propensity_scores"]) == len(df)
    assert (fit.extra["propensity_scores"] >= 0).all()
    assert (fit.extra["propensity_scores"] <= 1).all()


# ----------------------------------------------------------------------
# psm_predict
# ----------------------------------------------------------------------


def test_psm_predict_broadcasts_ate_estimate():
    df, _ = _confounded_treatment_data(n=5000)
    fit = reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    preds = reference.psm_predict(fit, df)
    assert preds.shape == (len(df),)
    assert (preds == fit.ate_estimate).all()


def test_psm_predict_without_ate_estimate_raises():
    fake_fit = FitResult(coefficients={}, raw_model=None, ate_estimate=None)
    with pytest.raises(ValueError, match="no ate_estimate"):
        reference.psm_predict(fake_fit, None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_psm_fit_all_treated_raises_clear_error():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, _ = gen.generate()
    df = df.copy()
    df["treat"] = 1
    with pytest.raises(ValueError, match="requires both treated and control units"):
        reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])


def test_psm_fit_all_control_raises_clear_error():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, _ = gen.generate()
    df = df.copy()
    df["treat"] = 0
    with pytest.raises(ValueError, match="requires both treated and control units"):
        reference.psm_fit(df, outcome="y", treatment="treat", covariates=["x1"])


# ----------------------------------------------------------------------
# Integration with compare_models
# ----------------------------------------------------------------------


def test_psm_outperforms_naive_diff_in_means_via_compare_models():
    from spuriosity import compare_models

    df, true_ate = _confounded_treatment_data()
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()  # placeholder truth object just for compare_models' signature

    def naive_fit(data):
        naive_effect = float(data[data.treat == 1]["y"].mean() - data[data.treat == 0]["y"].mean())
        return FitResult(coefficients={"treat": naive_effect}, ate_estimate=naive_effect)

    def naive_predict(fit_result, data):
        return np.full(len(data), fit_result.ate_estimate)

    # Build a truth object whose true_coefficients includes the real ATE for comparison.
    from spuriosity import GroundTruth

    truth_with_ate = GroundTruth(true_coefficients={"treat": true_ate})

    results = compare_models(
        data=df,
        truth=truth_with_ate,
        models={
            "naive": (naive_fit, naive_predict),
            "PSM": (reference.psm_fit, reference.psm_predict),
        },
        fit_kwargs_per_model={
            "naive": {},
            "PSM": {"outcome": "y", "treatment": "treat", "covariates": ["x1"]},
        },
    )
    table = results.ranked_table(by="coef_rmse")
    assert table.iloc[0]["model"] == "PSM"
