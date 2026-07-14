"""Tests for spuriosity.pathologies.Heteroskedasticity and Multicollinearity
(isolated, non-generator logic)."""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity.ground_truth import HeteroskedasticityInfo, MulticollinearityInfo
from spuriosity.pathologies import Heteroskedasticity, Multicollinearity


# ----------------------------------------------------------------------
# Heteroskedasticity
# ----------------------------------------------------------------------


def test_compute_noise_multiplier_basic_formula():
    h = Heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
    x1 = np.array([0.0, 1.0, 2.0, -2.0])
    result = h.compute_noise_multiplier(x1)
    np.testing.assert_allclose(result, [1.0, 1.5, 3.0, 3.0])


def test_compute_noise_multiplier_constant_formula_broadcasts():
    h = Heteroskedasticity(feature="x1", formula="2.0")
    x1 = np.array([0.0, 1.0, 2.0])
    result = h.compute_noise_multiplier(x1)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [2.0, 2.0, 2.0])


def test_compute_noise_multiplier_clamps_negative_to_zero():
    h = Heteroskedasticity(feature="x1", formula="x1")
    result = h.compute_noise_multiplier(np.array([-2.0, -1.0, 0.0, 1.0]))
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0])


def test_compute_noise_multiplier_invalid_formula_raises_clear_error():
    h = Heteroskedasticity(feature="x1", formula="1 + nonexistent_var")
    with pytest.raises(ValueError, match="Failed to evaluate heteroskedasticity formula"):
        h.compute_noise_multiplier(np.array([1.0, 2.0]))


def test_heteroskedasticity_ground_truth_contribution():
    h = Heteroskedasticity(feature="x1", formula="1 + 0.5*x1**2")
    contrib = h.ground_truth_contribution()
    assert contrib == {
        "heteroskedasticity": [HeteroskedasticityInfo(feature="x1", formula="1 + 0.5*x1**2")]
    }


# ----------------------------------------------------------------------
# Multicollinearity
# ----------------------------------------------------------------------


def test_construction_rejects_correlation_of_one():
    with pytest.raises(ValueError, match="disallowed"):
        Multicollinearity(feature="x2", correlated_with="x1", correlation=1.0)


def test_construction_rejects_negative_correlation():
    with pytest.raises(ValueError):
        Multicollinearity(feature="x2", correlated_with="x1", correlation=-0.1)


def test_construction_accepts_correlation_zero():
    # Boundary: correlation=0 should be valid (independent feature)
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.0)
    assert mc.correlation == 0.0


def test_generate_feature_achieves_target_correlation_at_scale():
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=1_000_000)
    x2 = mc.generate_feature(x1, rng)
    actual_corr = np.corrcoef(x1, x2)[0, 1]
    assert actual_corr == pytest.approx(0.9, abs=0.005)


def test_generate_feature_correlation_zero_gives_independent_feature():
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.0)
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=1_000_000)
    x2 = mc.generate_feature(x1, rng)
    actual_corr = np.corrcoef(x1, x2)[0, 1]
    assert actual_corr == pytest.approx(0.0, abs=0.01)


def test_generate_feature_has_approximately_unit_variance():
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.7)
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=5.0, scale=3.0, size=1_000_000)  # non-standard scale on purpose
    x2 = mc.generate_feature(x1, rng)
    assert x2.std() == pytest.approx(1.0, abs=0.01)


def test_generate_feature_zero_variance_correlated_with_raises():
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.5)
    rng = np.random.default_rng(42)
    x1_constant = np.full(1000, 5.0)  # zero variance
    with pytest.raises(ValueError, match="zero variance"):
        mc.generate_feature(x1_constant, rng)


def test_generate_feature_reproducible_with_same_rng_state():
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.8)
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result_a = mc.generate_feature(x1, np.random.default_rng(1))
    result_b = mc.generate_feature(x1, np.random.default_rng(1))
    np.testing.assert_allclose(result_a, result_b)


def test_multicollinearity_ground_truth_contribution():
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=0.9)
    contrib = mc.ground_truth_contribution()
    assert contrib == {
        "multicollinearity": [
            MulticollinearityInfo(feature="x2", correlated_with="x1", target_correlation=0.9)
        ]
    }


def test_implied_vif_formula_matches_actual_vif():
    """VIF = 1/(1-rho^2) for a two-feature case; verify the closed-form
    prediction against a real statsmodels VIF computation."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    rho = 0.85
    mc = Multicollinearity(feature="x2", correlated_with="x1", correlation=rho)
    rng = np.random.default_rng(42)
    n = 200_000
    x1 = rng.normal(size=n)
    x2 = mc.generate_feature(x1, rng)

    import pandas as pd

    X = pd.DataFrame({"const": 1.0, "x1": x1, "x2": x2})
    vif_x1 = variance_inflation_factor(X.values, 1)
    vif_x2 = variance_inflation_factor(X.values, 2)

    predicted_vif = 1 / (1 - rho**2)
    assert vif_x1 == pytest.approx(predicted_vif, abs=0.1)
    assert vif_x2 == pytest.approx(predicted_vif, abs=0.1)
