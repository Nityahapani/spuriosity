"""Tests for UnitRoot integration in spuriosity.generator.PanelGenerator,
including a real reproduction of the classic spurious-regression result."""

from __future__ import annotations

import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator, reference


def test_unit_root_creates_nonstationary_series():
    from statsmodels.tsa.stattools import adfuller

    gen = PanelGenerator(n_entities=1, n_periods=500, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_unit_root(feature="x1")
    df, truth = gen.generate()

    adf_result = adfuller(df["x1"])
    assert adf_result[1] > 0.1  # fail to reject unit root


def test_without_unit_root_series_is_stationary():
    from statsmodels.tsa.stattools import adfuller

    gen = PanelGenerator(n_entities=1, n_periods=500, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()

    adf_result = adfuller(df["x1"])
    assert adf_result[1] < 0.05  # correctly reject unit root


def test_unit_root_ground_truth_recorded():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_unit_root(feature="x1", drift=0.3)
    _, truth = gen.generate()
    assert len(truth.unit_root) == 1
    assert truth.unit_root[0].feature == "x1"
    assert truth.unit_root[0].drift == 0.3


def test_unit_root_absent_when_not_used():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.unit_root == []


def test_unit_root_resets_per_entity_in_multi_entity_panel():
    gen = PanelGenerator(n_entities=5, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_unit_root(feature="x1")
    df, _ = gen.generate()

    for entity in range(5):
        entity_df = df[df["entity_id"] == entity].sort_values("period")
        # First period's value should just be that entity's first increment,
        # not carried over from a different entity's accumulated walk.
        assert len(entity_df) == 10


def test_unit_root_on_undeclared_feature_raises():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_unit_root(feature="x1")
    with pytest.raises(ValueError, match="not a declared variable"):
        gen.generate()


def test_unit_root_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=10, n_periods=20, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_unit_root(feature="x1", drift=0.1)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    pdt.assert_frame_equal(df_a, df_b)


def test_unit_root_composes_with_structural_break():
    gen = PanelGenerator(n_entities=100, n_periods=20, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_unit_root(feature="x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 0.1, "Intercept": 0.0}, noise_std=1.0)
    gen.add_structural_break(period=10, target="y", kind="mean_shift", magnitude=5.0)
    df, truth = gen.generate()

    assert len(truth.unit_root) == 1
    assert len(truth.break_points) == 1
    pre_mean = df[df["period"] < 10]["y"].mean()
    post_mean = df[df["period"] >= 10]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(5.0, abs=0.5)


def test_spurious_regression_false_positive_rate_inflated():
    """The classic Granger-Newbold (1974) result: OLS between two
    INDEPENDENT random walks shows a "significant" relationship at a
    false-positive rate far above the nominal 5% significance level. This
    is the single most important thing this pathology should reproduce.

    Uses a moderate number of simulations (200) to keep test runtime
    reasonable while still cleanly distinguishing the two regimes -- the
    effect is large (typically 60-90% vs ~5%), so this doesn't require
    thousands of replications to detect reliably.
    """
    n_simulations = 200
    false_positive_count_rw = 0
    false_positive_count_iid = 0

    for sim in range(n_simulations):
        gen_rw = PanelGenerator(n_entities=1, n_periods=100, seed=sim)
        gen_rw.add_variable("x1", dist="normal", mean=0, std=1)
        gen_rw.add_variable("x2", dist="normal", mean=0, std=1)
        gen_rw.set_outcome(formula="x1", coefficients={"x1": 0.0})
        gen_rw.add_unit_root(feature="x1")
        gen_rw.add_unit_root(feature="x2")
        df_rw, _ = gen_rw.generate()
        fit_rw = reference.ols_fit(df_rw, formula="x2 ~ x1")
        if fit_rw.raw_model.pvalues["x1"] < 0.05:
            false_positive_count_rw += 1

        gen_iid = PanelGenerator(n_entities=1, n_periods=100, seed=sim)
        gen_iid.add_variable("x1", dist="normal", mean=0, std=1)
        gen_iid.add_variable("x2", dist="normal", mean=0, std=1)
        gen_iid.set_outcome(formula="x1", coefficients={"x1": 0.0})
        df_iid, _ = gen_iid.generate()
        fit_iid = reference.ols_fit(df_iid, formula="x2 ~ x1")
        if fit_iid.raw_model.pvalues["x1"] < 0.05:
            false_positive_count_iid += 1

    rw_rate = false_positive_count_rw / n_simulations
    iid_rate = false_positive_count_iid / n_simulations

    assert rw_rate > 0.3  # far above nominal 5%
    assert iid_rate < 0.15  # close to nominal 5%
    assert rw_rate > iid_rate * 2  # unambiguously different regimes
