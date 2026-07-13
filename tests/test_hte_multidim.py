"""Tests for multi-dimensional HTE support in spuriosity.hte.HTE.

Covers backward compatibility with v1's single-dimension API (which must
remain unchanged in call signature: cate_fn() -> Callable[[float], float],
called positionally) alongside the new multi-dimensional case.
"""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity.hte import HTE


# ----------------------------------------------------------------------
# Backward compatibility: single modifier as a string (v1 behavior)
# ----------------------------------------------------------------------


def test_single_string_modifier_is_not_multi_dim():
    h = HTE(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    assert h.is_multi_dim is False
    assert h.modifiers == ["x1"]
    assert h.modifier == "x1"


def test_single_dim_cate_fn_called_positionally():
    h = HTE(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    f = h.cate_fn()
    assert f(0.0) == pytest.approx(3.0)
    assert f(2.0) == pytest.approx(6.0)


def test_single_dim_evaluate_on_column_still_works():
    """The v1 singular evaluate_on_column(array) method must keep working
    unchanged for single-dimension HTEs."""
    h = HTE(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    x1 = np.array([0.0, 1.0, 2.0, -1.0])
    effects = h.evaluate_on_column(x1)
    np.testing.assert_allclose(effects, [3.0, 4.5, 6.0, 1.5])


# ----------------------------------------------------------------------
# Length-1 list is treated identically to a bare string
# ----------------------------------------------------------------------


def test_length_one_list_modifier_treated_as_single_dim():
    h = HTE(treatment="treat", modifier=["x1"], formula="3 + 1.5*x1")
    assert h.is_multi_dim is False
    assert h.modifier == "x1"
    f = h.cate_fn()
    assert f(2.0) == pytest.approx(6.0)


# ----------------------------------------------------------------------
# Multi-dimensional case
# ----------------------------------------------------------------------


def test_multi_dim_modifier_list():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    assert h.is_multi_dim is True
    assert h.modifiers == ["x1", "x2"]


def test_multi_dim_modifier_accessor_raises():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    with pytest.raises(AttributeError, match="use `.modifiers`"):
        h.modifier


def test_multi_dim_cate_fn_called_with_kwargs():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    f = h.cate_fn()
    assert f(x1=0.0, x2=0.0) == pytest.approx(3.0)
    assert f(x1=2.0, x2=1.0) == pytest.approx(5.5)


def test_multi_dim_cate_fn_called_positionally_raises_typeerror():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    f = h.cate_fn()
    with pytest.raises(TypeError):
        f(1.0, 2.0)


def test_multi_dim_cate_fn_missing_kwarg_raises_clear_error():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    f = h.cate_fn()
    with pytest.raises(TypeError, match="missing required keyword argument"):
        f(x1=1.0)


def test_multi_dim_cate_fn_extra_kwarg_raises_clear_error():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    f = h.cate_fn()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        f(x1=1.0, x2=2.0, x3=3.0)


def test_multi_dim_evaluate_on_columns():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    result = h.evaluate_on_columns({"x1": np.array([0.0, 2.0]), "x2": np.array([0.0, 1.0])})
    np.testing.assert_allclose(result, [3.0, 5.5])


def test_multi_dim_evaluate_on_columns_matches_cate_fn_pointwise():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="2 - 0.5*x1*x2")
    f = h.cate_fn()
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([1.0, -1.0, 3.0])
    vectorized = h.evaluate_on_columns({"x1": x1, "x2": x2})
    pointwise = np.array([f(x1=a, x2=b) for a, b in zip(x1, x2)])
    np.testing.assert_allclose(vectorized, pointwise)


def test_multi_dim_evaluate_on_column_singular_raises():
    """The legacy singular evaluate_on_column(array) method is
    single-dimension-only; using it on a multi-dim HTE should fail
    clearly, directing the caller to the plural method."""
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    with pytest.raises(AttributeError, match="evaluate_on_columns"):
        h.evaluate_on_column(np.array([1.0, 2.0]))


def test_evaluate_on_columns_missing_modifier_array_raises():
    h = HTE(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    with pytest.raises(ValueError, match="missing arrays for modifier"):
        h.evaluate_on_columns({"x1": np.array([1.0])})


def test_three_dimensional_modifier():
    """Sanity check beyond 2D -- the implementation should generalize to
    any number of modifiers, not just special-case 2."""
    h = HTE(treatment="treat", modifier=["x1", "x2", "x3"], formula="x1 + 2*x2 + 3*x3")
    f = h.cate_fn()
    assert f(x1=1.0, x2=1.0, x3=1.0) == pytest.approx(6.0)
    result = h.evaluate_on_columns(
        {"x1": np.array([1.0]), "x2": np.array([1.0]), "x3": np.array([1.0])}
    )
    np.testing.assert_allclose(result, [6.0])


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


def test_empty_modifier_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        HTE(treatment="treat", modifier=[], formula="1.0")
