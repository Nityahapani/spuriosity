"""Tests for spuriosity.pathologies.Confounder (isolated, non-generator logic)."""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity.pathologies import Confounder


def test_draw_and_apply_shifts_feature_and_outcome_by_same_u():
    conf = Confounder(feature="x1", outcome="y", strength=0.6)
    rng = np.random.default_rng(0)
    feature = np.zeros(1000)
    outcome_mean = np.zeros(1000)

    new_feature, new_outcome_mean, u = conf.draw_and_apply(feature, outcome_mean, rng)

    np.testing.assert_allclose(new_feature, 0.6 * u)
    np.testing.assert_allclose(new_outcome_mean, 0.6 * u)


def test_draw_and_apply_u_is_standard_normal():
    conf = Confounder(feature="x1", outcome="y", strength=0.6)
    rng = np.random.default_rng(0)
    n = 500_000
    _, _, u = conf.draw_and_apply(np.zeros(n), np.zeros(n), rng)
    assert u.mean() == pytest.approx(0.0, abs=0.01)
    assert u.std() == pytest.approx(1.0, abs=0.01)


def test_predicted_naive_bias_formula():
    conf = Confounder(feature="x1", outcome="y", strength=0.6)
    # bias = strength^2 / (feature_std^2 + strength^2)
    expected = 0.6**2 / (1.0**2 + 0.6**2)
    assert conf.predicted_naive_bias(feature_std=1.0) == pytest.approx(expected)


def test_predicted_naive_bias_scales_with_feature_std():
    conf = Confounder(feature="x1", outcome="y", strength=0.6)
    bias_std1 = conf.predicted_naive_bias(feature_std=1.0)
    bias_std2 = conf.predicted_naive_bias(feature_std=2.0)
    # larger feature variance dilutes the relative confounding influence
    assert bias_std2 < bias_std1


def test_predicted_naive_bias_zero_strength_gives_zero_bias():
    conf = Confounder(feature="x1", outcome="y", strength=0.0)
    assert conf.predicted_naive_bias(feature_std=1.0) == 0.0


def test_ground_truth_contribution():
    conf = Confounder(feature="x1", outcome="y", strength=0.6)
    contrib = conf.ground_truth_contribution()
    assert contrib == {"confounding_strength": {"x1": 0.6}}


def test_simulated_ovb_matches_predicted_bias():
    """End-to-end statistical check of the confounder mechanism in
    isolation, mirroring exactly how PanelGenerator applies it: the
    confounder shifts the feature *before* the outcome's own DGP reads it,
    and separately adds its contribution to the outcome mean."""
    rng = np.random.default_rng(0)
    n = 500_000
    strength = 0.6
    true_beta = 2.0

    conf = Confounder(feature="x1", outcome="y", strength=strength)
    x1_base = rng.normal(size=n)

    # Mirror PanelGenerator.generate(): confounder is applied to the raw
    # feature and to a zero-initialized outcome contribution BEFORE the
    # outcome's own formula/fn reads the (now-confounded) feature.
    x1_confounded, confounder_contribution, _ = conf.draw_and_apply(
        x1_base, np.zeros(n), rng
    )
    outcome_mean = true_beta * x1_confounded + confounder_contribution
    y = outcome_mean + rng.normal(scale=0.01, size=n)

    naive_slope = np.polyfit(x1_confounded, y, 1)[0]
    predicted_bias = conf.predicted_naive_bias(feature_std=1.0)

    assert (naive_slope - true_beta) == pytest.approx(predicted_bias, abs=0.02)
