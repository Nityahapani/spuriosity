"""Tests for spuriosity.hte.HTE (isolated, non-generator logic)."""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity.hte import HTE


def test_cate_fn_scalar_evaluation():
    h = HTE(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    f = h.cate_fn()
    assert f(0.0) == pytest.approx(3.0)
    assert f(2.0) == pytest.approx(6.0)
    assert f(-2.0) == pytest.approx(0.0)


def test_cate_fn_returns_plain_float():
    h = HTE(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    result = h.cate_fn()(1.0)
    assert isinstance(result, float)


def test_evaluate_on_column_vectorized():
    h = HTE(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    x1 = np.array([0.0, 1.0, 2.0, -1.0])
    effects = h.evaluate_on_column(x1)
    np.testing.assert_allclose(effects, [3.0, 4.5, 6.0, 1.5])


def test_evaluate_on_column_matches_cate_fn_pointwise():
    h = HTE(treatment="treat", modifier="x1", formula="2 - 0.5*x1**2")
    f = h.cate_fn()
    x1 = np.array([0.0, 1.0, 2.0, -1.0, 3.0])
    vectorized = h.evaluate_on_column(x1)
    pointwise = np.array([f(v) for v in x1])
    np.testing.assert_allclose(vectorized, pointwise)


def test_evaluate_on_column_constant_formula_broadcasts():
    """A formula with no dependence on the modifier (a constant effect)
    should still broadcast to the full array shape rather than returning
    a bare scalar."""
    h = HTE(treatment="treat", modifier="x1", formula="5.0")
    x1 = np.array([0.0, 1.0, 2.0])
    effects = h.evaluate_on_column(x1)
    assert effects.shape == (3,)
    np.testing.assert_allclose(effects, [5.0, 5.0, 5.0])


def test_evaluate_on_column_invalid_formula_raises_clear_error():
    h = HTE(treatment="treat", modifier="x1", formula="3 + nonexistent_var")
    x1 = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="Failed to evaluate HTE formula"):
        h.evaluate_on_column(x1)
