"""Tests for custom metric_registry integration in StressTest and
compare_models (v2 metrics registration API)."""

from __future__ import annotations

import pytest

from spuriosity import PanelGenerator, StressTest, reference
from spuriosity.metrics import MetricContext, MetricRegistry, default_registry
from spuriosity.stress_test import compare_models


def _sign_correctness(ctx: MetricContext):
    true = ctx.truth.true_coefficients
    fitted = ctx.fitted_coefficients
    shared = [k for k in true if k in fitted and true[k] != 0]
    if not shared:
        return None
    correct = sum(1 for k in shared if (fitted[k] > 0) == (true[k] > 0))
    return correct / len(shared)


def _basic_data(n_entities: int = 50_000, seed: int = 42):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.5)
    return gen.generate()


def test_stress_test_default_registry_used_when_none_supplied():
    df, truth = _basic_data(n_entities=1000)
    test = StressTest(truth)
    assert test.metric_registry is default_registry


def test_stress_test_custom_registry_is_used():
    df, truth = _basic_data()
    custom = default_registry.copy()
    custom.register("sign_correctness", _sign_correctness)

    test = StressTest(truth, metric_registry=custom)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"}, model_name="OLS",
    )
    assert "sign_correctness" in report.metrics
    assert report.metrics["sign_correctness"] == pytest.approx(1.0)


def test_stress_test_fresh_registry_excludes_builtin_metrics():
    """A from-scratch MetricRegistry() (not default_registry.copy()) should
    NOT compute coef_rmse etc. unless explicitly registered -- confirms
    metric_registry genuinely replaces, not merges with, the defaults."""
    df, truth = _basic_data(n_entities=1000)
    fresh = MetricRegistry()
    fresh.register("sign_correctness", _sign_correctness)

    test = StressTest(truth, metric_registry=fresh)
    report = test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"},
    )
    assert "coef_rmse" not in report.metrics
    assert "sign_correctness" in report.metrics


def test_compare_models_default_registry_when_none_supplied():
    df, truth = _basic_data(n_entities=1000)
    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1"}},
    )
    assert "coef_rmse" in results.reports["OLS"].metrics


def test_compare_models_custom_registry_threads_through_all_models():
    df, truth = _basic_data()
    custom = default_registry.copy()
    custom.register("sign_correctness", _sign_correctness)

    results = compare_models(
        data=df, truth=truth,
        models={
            "OLS": (reference.ols_fit, reference.ols_predict),
            "sklearn": (reference.sklearn_lr_fit, reference.sklearn_lr_predict),
        },
        fit_kwargs_per_model={
            "OLS": {"formula": "y ~ x1"},
            "sklearn": {"features": ["x1"], "target": "y"},
        },
        metric_registry=custom,
    )
    assert "sign_correctness" in results.reports["OLS"].metrics
    assert "sign_correctness" in results.reports["sklearn"].metrics


def test_default_registry_not_mutated_by_stress_test_usage():
    """Using StressTest with the default registry should never mutate the
    shared module-level default_registry object."""
    original_names = set(default_registry.names())
    df, truth = _basic_data(n_entities=1000)
    test = StressTest(truth)
    test.evaluate(
        fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df,
        fit_kwargs={"formula": "y ~ x1"},
    )
    assert set(default_registry.names()) == original_names
