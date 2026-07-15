"""Tests for spuriosity.pathologies.Endogeneity (isolated, non-generator
logic), including the hand-rolled first-stage F-statistic helper."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from spuriosity.pathologies import Endogeneity, _first_stage_f_stat


# ----------------------------------------------------------------------
# _first_stage_f_stat helper
# ----------------------------------------------------------------------


def test_first_stage_f_stat_matches_statsmodels():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n) * 0.5

    my_f = _first_stage_f_stat(x, y)
    X = sm.add_constant(x)
    sm_model = sm.OLS(y, X).fit()
    assert my_f == pytest.approx(sm_model.fvalue, rel=1e-6)


def test_first_stage_f_stat_large_for_strong_relationship():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = 5.0 * x + rng.normal(size=n) * 0.1
    assert _first_stage_f_stat(x, y) > 1000


def test_first_stage_f_stat_small_for_no_relationship():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # independent of x
    f = _first_stage_f_stat(x, y)
    assert f < 10  # should not be significant


# ----------------------------------------------------------------------
# Endogeneity construction
# ----------------------------------------------------------------------


def test_construction_rejects_negative_first_stage_noise_std():
    with pytest.raises(ValueError):
        Endogeneity(
            feature="x1", instrument="z", instrument_strength=1.0,
            endogeneity_strength=1.0, first_stage_noise_std=-1.0,
        )


# ----------------------------------------------------------------------
# Endogeneity.generate
# ----------------------------------------------------------------------


def test_generate_returns_correct_shapes():
    endo = Endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    rng = np.random.default_rng(0)
    instrument_vals, feature_vals, contribution = endo.generate(1000, rng)
    assert instrument_vals.shape == (1000,)
    assert feature_vals.shape == (1000,)
    assert contribution.shape == (1000,)


def test_generate_instrument_is_standard_normal():
    endo = Endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    rng = np.random.default_rng(0)
    instrument_vals, _, _ = endo.generate(500_000, rng)
    assert instrument_vals.mean() == pytest.approx(0.0, abs=0.01)
    assert instrument_vals.std() == pytest.approx(1.0, abs=0.01)


def test_generate_feature_correlates_with_instrument_when_strong():
    endo = Endogeneity(
        feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=0.1,
        first_stage_noise_std=0.1,
    )
    rng = np.random.default_rng(0)
    instrument_vals, feature_vals, _ = endo.generate(500_000, rng)
    corr = np.corrcoef(instrument_vals, feature_vals)[0, 1]
    assert corr > 0.9  # strong instrument, weak endogeneity, low noise -> high correlation


def test_generate_realized_f_stat_populated_after_call():
    endo = Endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    assert endo._realized_f_stat is None
    rng = np.random.default_rng(0)
    endo.generate(10_000, rng)
    assert endo._realized_f_stat is not None
    assert endo._realized_f_stat > 0


def test_generate_weak_instrument_gives_low_f_stat():
    endo = Endogeneity(
        feature="x1", instrument="z", instrument_strength=0.01, endogeneity_strength=1.0
    )
    rng = np.random.default_rng(0)
    endo.generate(50_000, rng)
    assert endo._realized_f_stat < 10  # below Stock-Yogo rule of thumb


def test_generate_strong_instrument_gives_high_f_stat():
    endo = Endogeneity(
        feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=0.5
    )
    rng = np.random.default_rng(0)
    endo.generate(50_000, rng)
    assert endo._realized_f_stat > 100


# ----------------------------------------------------------------------
# ground_truth_contribution
# ----------------------------------------------------------------------


def test_ground_truth_contribution_shape():
    endo = Endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=0.5)
    rng = np.random.default_rng(0)
    endo.generate(1000, rng)
    contrib = endo.ground_truth_contribution()
    info = contrib["endogeneity"][0]
    assert info.feature == "x1"
    assert info.instrument == "z"
    assert info.instrument_strength == 1.0
    assert info.endogeneity_strength == 0.5
    assert info.realized_first_stage_f_stat is not None


def test_ground_truth_contribution_before_generate_has_none_f_stat():
    """Unlike MeasurementError (which raises if called before apply()),
    Endogeneity's ground_truth_contribution can be called before generate()
    -- it just reports realized_first_stage_f_stat=None in that case,
    since instrument_strength/endogeneity_strength are known at
    construction time regardless."""
    endo = Endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=0.5)
    contrib = endo.ground_truth_contribution()
    assert contrib["endogeneity"][0].realized_first_stage_f_stat is None


# ----------------------------------------------------------------------
# Simulated end-to-end bias check (isolated from PanelGenerator)
# ----------------------------------------------------------------------


def test_simulated_naive_ols_biased_2sls_unbiased():
    rng = np.random.default_rng(0)
    n = 1_000_000
    true_beta = 2.0

    endo = Endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    instrument_vals, feature_vals, contribution = endo.generate(n, rng)
    y = true_beta * feature_vals + contribution + rng.normal(scale=0.01, size=n)

    naive_beta = np.polyfit(feature_vals, y, 1)[0]
    assert abs(naive_beta - true_beta) > 0.2  # meaningfully biased

    # manual 2SLS: first stage, then second stage on fitted values
    first_stage = np.polyfit(instrument_vals, feature_vals, 1)
    feature_hat = first_stage[0] * instrument_vals + first_stage[1]
    second_stage = np.polyfit(feature_hat, y, 1)
    assert second_stage[0] == pytest.approx(true_beta, abs=0.02)
