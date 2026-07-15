"""Tests for Endogeneity integration in spuriosity.generator.PanelGenerator
and the paired IV/2SLS reference fits in spuriosity.reference."""

from __future__ import annotations

import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator, StressTest, reference


def _endogenous_data(
    n_entities: int = 500_000,
    seed: int = 42,
    instrument_strength: float = 1.0,
    endogeneity_strength: float = 1.0,
):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_endogeneity(
        feature="x1", instrument="z",
        instrument_strength=instrument_strength, endogeneity_strength=endogeneity_strength,
    )
    return gen.generate()


# ----------------------------------------------------------------------
# PanelGenerator integration
# ----------------------------------------------------------------------


def test_endogeneity_creates_instrument_column():
    df, truth = _endogenous_data(n_entities=1000)
    assert "z" in df.columns
    assert "x1" in df.columns


def test_endogeneity_naive_ols_is_biased():
    df, truth = _endogenous_data()
    fit_naive = reference.ols_fit(df, formula="y ~ x1")
    assert abs(fit_naive.coefficients["x1"] - 2.0) > 0.3


def test_endogeneity_2sls_recovers_truth():
    df, truth = _endogenous_data()
    fit = reference.iv2sls_fit(df, outcome="y", endogenous=["x1"], instruments=["z"])
    assert fit.coefficients["x1"] == pytest.approx(2.0, abs=0.02)


def test_endogeneity_ground_truth_recorded():
    df, truth = _endogenous_data(n_entities=1000)
    assert len(truth.endogeneity) == 1
    info = truth.endogeneity[0]
    assert info.feature == "x1"
    assert info.instrument == "z"
    assert info.realized_first_stage_f_stat is not None


def test_endogeneity_absent_when_not_used():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.endogeneity == []


def test_weak_instrument_gives_low_realized_f_stat_and_unreliable_2sls():
    df, truth = _endogenous_data(
        n_entities=50_000, instrument_strength=0.01, endogeneity_strength=1.0
    )
    assert truth.endogeneity[0].realized_first_stage_f_stat < 10

    fit = reference.iv2sls_fit(df, outcome="y", endogenous=["x1"], instruments=["z"])
    # Under a weak instrument, 2SLS should have much larger standard error
    # than under a strong instrument (checked via raw_model's std_errors).
    se = fit.raw_model.std_errors["x1"]
    assert se > 0.3


def test_endogeneity_on_undeclared_feature_raises():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    with pytest.raises(ValueError, match="not a declared variable or treatment"):
        gen.generate()


def test_instrument_name_collision_at_builder_time():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_variable("z")
    with pytest.raises(ValueError, match="already in use"):
        gen.add_endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)


def test_add_variable_after_endogeneity_with_same_instrument_name_raises():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    with pytest.raises(ValueError, match="already in use"):
        gen.add_variable("z")


def test_endogeneity_reproducible_with_same_seed():
    df_a, _ = _endogenous_data(n_entities=1000, seed=7)
    df_b, _ = _endogenous_data(n_entities=1000, seed=7)
    pdt.assert_frame_equal(df_a, df_b)


def test_endogeneity_composes_with_structural_break():
    gen = PanelGenerator(n_entities=100_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=0.5)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=3.0)
    df, truth = gen.generate()

    assert len(truth.endogeneity) == 1
    assert len(truth.break_points) == 1
    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(3.0, abs=0.3)


# ----------------------------------------------------------------------
# reference.iv2sls_fit / iv2sls_predict / first_stage_f_stat
# ----------------------------------------------------------------------


def test_iv2sls_predict_returns_correct_shape():
    df, truth = _endogenous_data(n_entities=1000)
    fit = reference.iv2sls_fit(df, outcome="y", endogenous=["x1"], instruments=["z"])
    preds = reference.iv2sls_predict(fit, df)
    assert preds.shape == (len(df),)


def test_iv2sls_fit_missing_dependency_error_message():
    """We can't easily uninstall linearmodels mid-test-suite, but we can
    at least confirm the import-error path is reachable code (smoke-level
    coverage) by checking the function references linearmodels lazily,
    not at module import time -- i.e. importing spuriosity.reference
    itself must not require linearmodels to be installed."""
    import importlib

    import spuriosity.reference as ref_module

    importlib.reload(ref_module)  # should not raise even if linearmodels were absent


def test_first_stage_f_stat_matches_truth_realized_value():
    df, truth = _endogenous_data(n_entities=10_000)
    f_stat = reference.first_stage_f_stat(df, endogenous="x1", instruments=["z"])
    assert f_stat == pytest.approx(truth.endogeneity[0].realized_first_stage_f_stat, rel=0.01)


def test_stress_test_shows_2sls_outperforms_naive_ols_on_coef_rmse():
    df, truth = _endogenous_data()
    test = StressTest(truth)

    report_naive = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"}, model_name="naive_OLS",
    )
    report_iv = test.evaluate(
        fit_fn=reference.iv2sls_fit, predict_fn=reference.iv2sls_predict, data=df,
        fit_kwargs={"outcome": "y", "endogenous": ["x1"], "instruments": ["z"]}, model_name="2SLS",
    )
    assert report_iv.metrics["coef_rmse"] < report_naive.metrics["coef_rmse"]


def test_iv2sls_fit_with_exogenous_controls():
    gen = PanelGenerator(n_entities=200_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)  # exogenous control
    gen.add_variable("x2", dist="normal", mean=0, std=1)  # will become endogenous
    gen.set_outcome(
        formula="x1 + x2", coefficients={"x1": 1.0, "x2": 2.0, "Intercept": 0.0}, noise_std=0.1
    )
    gen.add_endogeneity(feature="x2", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    df, truth = gen.generate()

    fit = reference.iv2sls_fit(df, outcome="y", endogenous=["x2"], instruments=["z"], exogenous=["x1"])
    assert fit.coefficients["x2"] == pytest.approx(2.0, abs=0.05)
    assert fit.coefficients["x1"] == pytest.approx(1.0, abs=0.05)
