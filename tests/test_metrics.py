"""Tests for spuriosity.metrics — the pluggable metrics registration API.

Includes regression tests verifying the built-in metrics produce identical
values to v1's hard-coded implementation, since this module is a refactor
of previously inline logic in StressTest.evaluate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spuriosity.ground_truth import BreakInfo, GroundTruth
from spuriosity.metrics import MetricContext, MetricRegistry, default_registry


def _ctx(
    truth: GroundTruth,
    fitted_coefficients: dict,
    data: pd.DataFrame | None = None,
    fit_fn=None,
    fit_kwargs: dict | None = None,
) -> MetricContext:
    return MetricContext(
        truth=truth,
        fitted_coefficients=fitted_coefficients,
        fit_result=None,
        data=data if data is not None else pd.DataFrame(),
        fit_fn=fit_fn if fit_fn is not None else (lambda *a, **k: None),
        fit_kwargs=fit_kwargs or {},
    )


# ----------------------------------------------------------------------
# MetricRegistry basics
# ----------------------------------------------------------------------


def test_register_and_run_single_float_metric():
    registry = MetricRegistry()
    registry.register("always_one", lambda ctx: 1.0)
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert results == {"always_one": 1.0}


def test_metric_returning_none_is_omitted_not_zero():
    registry = MetricRegistry()
    registry.register("never_applies", lambda ctx: None)
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert "never_applies" not in results


def test_register_overwrites_existing_name():
    registry = MetricRegistry()
    registry.register("m", lambda ctx: 1.0)
    registry.register("m", lambda ctx: 2.0)
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert results == {"m": 2.0}


def test_unregister_removes_metric():
    registry = MetricRegistry()
    registry.register("m", lambda ctx: 1.0)
    registry.unregister("m")
    assert "m" not in registry.names()


def test_unregister_missing_name_raises_keyerror():
    registry = MetricRegistry()
    with pytest.raises(KeyError):
        registry.unregister("nonexistent")


def test_names_lists_registered_metrics():
    registry = MetricRegistry()
    registry.register("a", lambda ctx: 1.0)
    registry.register("b", lambda ctx: 2.0)
    assert set(registry.names()) == {"a", "b"}


def test_copy_is_independent_of_original():
    registry = MetricRegistry()
    registry.register("a", lambda ctx: 1.0)
    copied = registry.copy()
    copied.register("b", lambda ctx: 2.0)
    assert "b" not in registry.names()
    assert "b" in copied.names()


def test_dict_returning_metric_namespaces_keys_under_metric_name():
    registry = MetricRegistry()
    registry.register("multi", lambda ctx: {"foo": 1.0, "bar": 2.0})
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert results == {"multi:foo": 1.0, "multi:bar": 2.0}


def test_dict_returning_metric_stores_matching_key_bare():
    """A dict entry whose key matches the metric's own registered name is
    stored bare (not double-namespaced as 'name:name') -- this is how
    confounding_bias produces both a bare aggregate and per-feature
    breakdown keys from one function."""
    registry = MetricRegistry()
    registry.register("agg", lambda ctx: {"x1": 0.5, "agg": 0.5})
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert results == {"agg:x1": 0.5, "agg": 0.5}


def test_dict_returning_metric_empty_dict_treated_as_inapplicable():
    registry = MetricRegistry()
    registry.register("multi", lambda ctx: {})
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert results == {}


def test_run_all_evaluates_every_registered_metric():
    registry = MetricRegistry()
    registry.register("a", lambda ctx: 1.0)
    registry.register("b", lambda ctx: None)
    registry.register("c", lambda ctx: 3.0)
    truth = GroundTruth(true_coefficients={})
    results = registry.run_all(_ctx(truth, {}))
    assert results == {"a": 1.0, "c": 3.0}


# ----------------------------------------------------------------------
# default_registry: built-in metrics (regression tests vs. v1 behavior)
# ----------------------------------------------------------------------


def test_default_registry_has_four_builtin_metrics():
    assert set(default_registry.names()) == {
        "coef_rmse",
        "confounding_bias",
        "break_detection_lag",
        "cate_rmse",
    }


def test_coef_rmse_matches_known_value():
    truth = GroundTruth(true_coefficients={"x1": 2.0, "x2": 0.5})
    fitted = {"x1": 2.1, "x2": 0.4}
    results = default_registry.run_all(_ctx(truth, fitted))
    # RMSE of [0.1, -0.1] = sqrt(mean([0.01, 0.01])) = 0.1
    assert results["coef_rmse"] == pytest.approx(0.1)


def test_coef_rmse_absent_when_true_coefficients_empty():
    truth = GroundTruth(true_coefficients={})
    results = default_registry.run_all(_ctx(truth, {"x1": 1.0}))
    assert "coef_rmse" not in results


def test_coef_rmse_absent_when_no_shared_keys():
    truth = GroundTruth(true_coefficients={"x1": 1.0})
    results = default_registry.run_all(_ctx(truth, {"totally_different": 5.0}))
    assert "coef_rmse" not in results


def test_confounding_bias_matches_known_value_and_produces_per_feature_key():
    truth = GroundTruth(
        true_coefficients={"x1": 2.0},
        confounding_strength={"x1": 0.6},
    )
    fitted = {"x1": 2.26}
    results = default_registry.run_all(_ctx(truth, fitted))
    assert results["confounding_bias:x1"] == pytest.approx(0.26)
    assert results["confounding_bias"] == pytest.approx(0.26)


def test_confounding_bias_absent_without_confounder():
    truth = GroundTruth(true_coefficients={"x1": 1.0})
    results = default_registry.run_all(_ctx(truth, {"x1": 1.0}))
    assert "confounding_bias" not in results


def test_confounding_bias_multiple_features_averaged():
    truth = GroundTruth(
        true_coefficients={"x1": 1.0, "x2": 1.0},
        confounding_strength={"x1": 0.5, "x2": 0.5},
    )
    fitted = {"x1": 1.2, "x2": 1.4}
    results = default_registry.run_all(_ctx(truth, fitted))
    assert results["confounding_bias:x1"] == pytest.approx(0.2)
    assert results["confounding_bias:x2"] == pytest.approx(0.4)
    assert results["confounding_bias"] == pytest.approx(0.3)  # mean(0.2, 0.4)


def test_break_detection_lag_absent_without_break_points():
    truth = GroundTruth(true_coefficients={"x1": 1.0})
    results = default_registry.run_all(_ctx(truth, {"x1": 1.0}))
    assert "break_detection_lag" not in results


def test_break_detection_lag_absent_without_formula_kwarg():
    truth = GroundTruth(
        true_coefficients={"x1": 1.0},
        break_points=[BreakInfo(period=5, target="y", kind="coefficient_shift", magnitude=2.0)],
    )
    results = default_registry.run_all(
        _ctx(truth, {"x1": 1.0}, fit_kwargs={})  # no formula= kwarg
    )
    assert "break_detection_lag" not in results


def test_break_detection_lag_absent_for_non_coefficient_shift_break():
    truth = GroundTruth(
        true_coefficients={"x1": 1.0},
        break_points=[BreakInfo(period=5, target="y", kind="mean_shift", magnitude=2.0)],
    )
    results = default_registry.run_all(
        _ctx(truth, {"x1": 1.0}, fit_kwargs={"formula": "y ~ x1"})
    )
    assert "break_detection_lag" not in results


def test_cate_rmse_always_absent_v1_placeholder():
    """cate_rmse is a defined slot but not yet implemented (deferred, see
    module docstring); it should never appear in results in v1/early v2."""
    truth = GroundTruth(true_coefficients={"x1": 1.0})
    results = default_registry.run_all(_ctx(truth, {"x1": 1.0}))
    assert "cate_rmse" not in results
