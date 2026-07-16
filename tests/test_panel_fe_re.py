"""Tests for spuriosity.reference panel FE/RE fits and hausman_test.

These construct the classic "entity effect correlated with regressor"
scenario by hand (spuriosity does not yet have a dedicated pathology for
injecting a time-invariant panel entity effect -- a natural follow-up),
then verify FE recovers the truth while RE is biased, and that the
Hausman test correctly flags this in the biased case and correctly does
not flag it when entity effects are genuinely uncorrelated with the
regressor.
"""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity import PanelGenerator, reference
from spuriosity.reference import FitResult


def _correlated_entity_effect_data(n_entities: int = 2000, n_periods: int = 10, seed: int = 42):
    """Entity effect correlated with x1 -- RE should be biased here."""
    rng = np.random.default_rng(seed)
    entity_effect = rng.normal(size=n_entities)

    gen = PanelGenerator(n_entities=n_entities, n_periods=n_periods, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.5)
    df, truth = gen.generate()

    df = df.copy()
    df["x1"] = df["x1"] + 0.8 * entity_effect[df["entity_id"]]
    df["y"] = df["y"] + entity_effect[df["entity_id"]]
    return df, truth


def _uncorrelated_entity_effect_data(n_entities: int = 2000, n_periods: int = 10, seed: int = 42):
    """Entity effect independent of x1 -- RE should be valid (unbiased,
    more efficient than FE) here."""
    rng = np.random.default_rng(seed)
    entity_effect = rng.normal(size=n_entities)

    gen = PanelGenerator(n_entities=n_entities, n_periods=n_periods, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.5)
    df, truth = gen.generate()

    df = df.copy()
    df["y"] = df["y"] + entity_effect[df["entity_id"]]
    return df, truth


# ----------------------------------------------------------------------
# panel_fe_fit
# ----------------------------------------------------------------------


def test_panel_fe_recovers_truth_even_with_correlated_entity_effect():
    df, truth = _correlated_entity_effect_data()
    fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    assert fit.coefficients["x1"] == pytest.approx(2.0, abs=0.1)


def test_panel_fe_predict_shape():
    df, truth = _correlated_entity_effect_data(n_entities=200)
    fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    preds = reference.panel_fe_predict(fit, df)
    assert preds.shape == (len(df),)


def test_panel_fe_has_no_intercept():
    df, truth = _correlated_entity_effect_data(n_entities=200)
    fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    assert "Intercept" not in fit.coefficients
    assert "const" not in fit.coefficients


# ----------------------------------------------------------------------
# panel_re_fit
# ----------------------------------------------------------------------


def test_panel_re_biased_when_entity_effect_correlates_with_regressor():
    df, truth = _correlated_entity_effect_data()
    fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    assert abs(fit.coefficients["x1"] - 2.0) > 0.05


def test_panel_re_unbiased_when_entity_effect_independent_of_regressor():
    df, truth = _uncorrelated_entity_effect_data()
    fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    assert fit.coefficients["x1"] == pytest.approx(2.0, abs=0.1)


def test_panel_re_predict_shape():
    df, truth = _uncorrelated_entity_effect_data(n_entities=200)
    fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    preds = reference.panel_re_predict(fit, df)
    assert preds.shape == (len(df),)


def test_panel_re_has_const_intercept():
    df, truth = _uncorrelated_entity_effect_data(n_entities=200)
    fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    assert "const" in fit.coefficients


# ----------------------------------------------------------------------
# hausman_test
# ----------------------------------------------------------------------


def test_hausman_rejects_re_when_entity_effect_correlates_with_regressor():
    df, truth = _correlated_entity_effect_data()
    fe_fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    re_fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    result = reference.hausman_test(fe_fit, re_fit)
    assert result["p_value"] < 0.05


def test_hausman_does_not_reject_re_when_valid():
    df, truth = _uncorrelated_entity_effect_data()
    fe_fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    re_fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    result = reference.hausman_test(fe_fit, re_fit)
    assert result["p_value"] > 0.1


def test_hausman_result_has_expected_keys():
    df, truth = _uncorrelated_entity_effect_data(n_entities=500)
    fe_fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    re_fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    result = reference.hausman_test(fe_fit, re_fit)
    assert set(result.keys()) == {"chi2", "dof", "p_value"}
    assert result["dof"] == 1.0


def test_hausman_raises_on_no_shared_coefficients():
    fake_fe = FitResult(coefficients={"a": 1.0}, raw_model=None)
    fake_re = FitResult(coefficients={"b": 1.0}, raw_model=None)
    with pytest.raises(ValueError, match="no coefficients shared"):
        reference.hausman_test(fake_fe, fake_re)


def test_hausman_excludes_re_only_intercept_from_comparison():
    """RE has a 'const' coefficient with no FE counterpart; the Hausman
    test should silently compare only shared coefficients rather than
    erroring or including a nonsensical comparison."""
    df, truth = _uncorrelated_entity_effect_data(n_entities=500)
    fe_fit = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    re_fit = reference.panel_re_fit(df, outcome="y", features=["x1"])
    assert "const" not in fe_fit.coefficients
    assert "const" in re_fit.coefficients
    result = reference.hausman_test(fe_fit, re_fit)
    assert result["dof"] == 1.0  # only x1 compared, not const


# ----------------------------------------------------------------------
# Integration with compare_models
# ----------------------------------------------------------------------


def test_fe_outranks_re_and_pooled_ols_in_compare_models():
    df, truth = _correlated_entity_effect_data()
    from spuriosity import compare_models

    results = compare_models(
        data=df,
        truth=truth,
        models={
            "FE": (reference.panel_fe_fit, reference.panel_fe_predict),
            "RE": (reference.panel_re_fit, reference.panel_re_predict),
            "pooled_OLS": (reference.ols_fit, reference.ols_predict),
        },
        fit_kwargs_per_model={
            "FE": {"outcome": "y", "features": ["x1"]},
            "RE": {"outcome": "y", "features": ["x1"]},
            "pooled_OLS": {"formula": "y ~ x1"},
        },
    )
    table = results.ranked_table(by="coef_rmse")
    assert table.iloc[0]["model"] == "FE"
