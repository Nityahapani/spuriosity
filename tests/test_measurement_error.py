"""Tests for spuriosity.pathologies.MeasurementError (isolated,
non-generator logic)."""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity.ground_truth import MeasurementErrorInfo
from spuriosity.pathologies import MeasurementError


def test_construction_rejects_negative_noise_std():
    with pytest.raises(ValueError):
        MeasurementError(feature="x1", noise_std=-1.0)


def test_construction_accepts_zero_noise_std():
    me = MeasurementError(feature="x1", noise_std=0.0)
    assert me.noise_std == 0.0


def test_apply_zero_noise_returns_unchanged_values():
    me = MeasurementError(feature="x1", noise_std=0.0)
    rng = np.random.default_rng(0)
    true_vals = np.array([1.0, 2.0, 3.0])
    observed = me.apply(true_vals, rng)
    np.testing.assert_allclose(observed, true_vals)


def test_apply_increases_variance():
    me = MeasurementError(feature="x1", noise_std=0.8)
    rng = np.random.default_rng(0)
    true_vals = rng.normal(scale=1.0, size=1_000_000)
    observed = me.apply(true_vals, rng)
    assert np.var(observed) > np.var(true_vals)


def test_apply_mean_unbiased():
    """Measurement error should not shift the mean, only inflate variance
    (classical, not systematic, measurement error)."""
    me = MeasurementError(feature="x1", noise_std=0.8)
    rng = np.random.default_rng(0)
    true_vals = rng.normal(loc=5.0, scale=1.0, size=1_000_000)
    observed = me.apply(true_vals, rng)
    assert observed.mean() == pytest.approx(true_vals.mean(), abs=0.01)


def test_ground_truth_contribution_before_apply_raises():
    me = MeasurementError(feature="x1", noise_std=0.5)
    with pytest.raises(RuntimeError, match="called before apply"):
        me.ground_truth_contribution()


def test_ground_truth_contribution_after_apply():
    me = MeasurementError(feature="x1", noise_std=0.8)
    rng = np.random.default_rng(0)
    true_vals = rng.normal(scale=1.0, size=1_000_000)
    me.apply(true_vals, rng)
    contrib = me.ground_truth_contribution()
    assert "measurement_error" in contrib
    info = contrib["measurement_error"][0]
    assert isinstance(info, MeasurementErrorInfo)
    assert info.feature == "x1"
    assert info.noise_std == 0.8


def test_realized_reliability_ratio_matches_closed_form():
    me = MeasurementError(feature="x1", noise_std=0.8)
    rng = np.random.default_rng(0)
    true_vals = rng.normal(scale=1.0, size=1_000_000)
    me.apply(true_vals, rng)
    info = me.ground_truth_contribution()["measurement_error"][0]

    true_variance = np.var(true_vals)
    predicted_ratio = true_variance / (true_variance + 0.8**2)
    assert info.reliability_ratio == pytest.approx(predicted_ratio, abs=1e-6)


def test_zero_noise_gives_reliability_ratio_one():
    me = MeasurementError(feature="x1", noise_std=0.0)
    rng = np.random.default_rng(0)
    true_vals = rng.normal(size=1000)
    me.apply(true_vals, rng)
    info = me.ground_truth_contribution()["measurement_error"][0]
    assert info.reliability_ratio == 1.0


def test_larger_noise_gives_lower_reliability_ratio():
    rng = np.random.default_rng(0)
    true_vals = rng.normal(scale=1.0, size=500_000)

    me_small = MeasurementError(feature="x1", noise_std=0.2)
    me_small.apply(true_vals, np.random.default_rng(1))
    ratio_small = me_small.ground_truth_contribution()["measurement_error"][0].reliability_ratio

    me_large = MeasurementError(feature="x1", noise_std=2.0)
    me_large.apply(true_vals, np.random.default_rng(1))
    ratio_large = me_large.ground_truth_contribution()["measurement_error"][0].reliability_ratio

    assert ratio_large < ratio_small


def test_simulated_attenuation_bias_matches_reliability_ratio():
    """End-to-end statistical check of the mechanism in isolation: a naive
    regression on the noisy observed feature should be attenuated by
    exactly the realized reliability ratio."""
    rng = np.random.default_rng(0)
    n = 1_000_000
    true_beta = 2.0

    me = MeasurementError(feature="x1", noise_std=0.8)
    x1_true = rng.normal(scale=1.0, size=n)
    y = true_beta * x1_true + rng.normal(scale=0.01, size=n)

    x1_observed = me.apply(x1_true, rng)
    info = me.ground_truth_contribution()["measurement_error"][0]

    naive_slope = np.polyfit(x1_observed, y, 1)[0]
    predicted_slope = true_beta * info.reliability_ratio

    assert naive_slope == pytest.approx(predicted_slope, abs=0.02)
