"""Tests for spuriosity.stress_test.compare_models / ComparisonReport."""

from __future__ import annotations

import pandas as pd

from spuriosity import PanelGenerator, reference
from spuriosity.stress_test import compare_models


def _confounded_data(n_entities: int = 200_000, seed: int = 42):
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_confounder(feature="x1", outcome="y", strength=0.5, observed=True)
    return gen.generate()


def test_compare_models_ranks_controlled_model_above_naive():
    df, truth = _confounded_data()
    results = compare_models(
        data=df,
        truth=truth,
        models={
            "naive_OLS": (reference.ols_fit, reference.ols_predict),
            "controlled_OLS": (reference.ols_fit, reference.ols_predict),
        },
        fit_kwargs_per_model={
            "naive_OLS": {"formula": "y ~ x1"},
            "controlled_OLS": {"formula": "y ~ x1 + _confounder_x1"},
        },
    )
    table = results.ranked_table(by="default_composite")
    assert table.iloc[0]["model"] == "controlled_OLS"
    assert table.iloc[0]["default_composite"] < table.iloc[1]["default_composite"]


def test_ranked_table_returns_dataframe():
    df, truth = _confounded_data(n_entities=1000)
    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1 + _confounder_x1"}},
    )
    table = results.ranked_table()
    assert isinstance(table, pd.DataFrame)
    assert "model" in table.columns


def test_ranked_table_by_single_metric():
    df, truth = _confounded_data(n_entities=1000)
    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1 + _confounder_x1"}},
    )
    table = results.ranked_table(by="coef_rmse")
    assert "coef_rmse" in table.columns
    assert list(table["model"]) == ["OLS"]


def test_ranked_table_excludes_models_missing_the_requested_metric():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()  # no confounder -> no confounding_bias metric

    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1"}},
    )
    table = results.ranked_table(by="confounding_bias")
    assert table.empty
    assert table.attrs["excluded_models"] == ["OLS"]


def test_custom_weights_override_defaults_but_keep_unspecified_at_default():
    df, truth = _confounded_data(n_entities=1000)
    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1 + _confounder_x1"}},
        weights={"coef_rmse": 10.0},
    )
    assert results.weights["coef_rmse"] == 10.0
    assert results.weights["cate_rmse"] == 1.0  # untouched default


def test_reports_accessible_individually_regardless_of_ranking():
    df, truth = _confounded_data(n_entities=1000)
    results = compare_models(
        data=df, truth=truth,
        models={
            "model_a": (reference.ols_fit, reference.ols_predict),
            "model_b": (reference.ols_fit, reference.ols_predict),
        },
        fit_kwargs_per_model={
            "model_a": {"formula": "y ~ x1"},
            "model_b": {"formula": "y ~ x1 + _confounder_x1"},
        },
    )
    assert set(results.reports.keys()) == {"model_a", "model_b"}
    assert "coef_rmse" in results.reports["model_a"].metrics
    assert "coef_rmse" in results.reports["model_b"].metrics


def test_compare_models_with_three_models_including_sklearn():
    df, truth = _confounded_data(n_entities=50_000)
    results = compare_models(
        data=df, truth=truth,
        models={
            "naive_OLS": (reference.ols_fit, reference.ols_predict),
            "controlled_OLS": (reference.ols_fit, reference.ols_predict),
            "controlled_sklearn": (reference.sklearn_lr_fit, reference.sklearn_lr_predict),
        },
        fit_kwargs_per_model={
            "naive_OLS": {"formula": "y ~ x1"},
            "controlled_OLS": {"formula": "y ~ x1 + _confounder_x1"},
            "controlled_sklearn": {"features": ["x1", "_confounder_x1"], "target": "y"},
        },
    )
    table = results.ranked_table(by="default_composite")
    assert len(table) == 3
    # both controlled models should outrank the naive one
    naive_rank = table[table["model"] == "naive_OLS"].index[0]
    controlled_ols_rank = table[table["model"] == "controlled_OLS"].index[0]
    controlled_sklearn_rank = table[table["model"] == "controlled_sklearn"].index[0]
    assert controlled_ols_rank < naive_rank
    assert controlled_sklearn_rank < naive_rank


def test_empty_models_dict_produces_empty_reports():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()
    results = compare_models(data=df, truth=truth, models={})
    assert results.reports == {}
    assert results.ranked_table().empty


def test_summary_does_not_raise_with_multiple_models():
    """`summary()` should print a human-readable report without raising,
    mirroring `StressTestReport.summary()` for API symmetry."""
    df, truth = _confounded_data(n_entities=2000)
    results = compare_models(
        data=df, truth=truth,
        models={
            "naive_OLS":       (reference.ols_fit, reference.ols_predict),
            "controlled_OLS":  (reference.ols_fit, reference.ols_predict),
        },
        fit_kwargs_per_model={
            "naive_OLS":      {"formula": "y ~ x1"},
            "controlled_OLS": {"formula": "y ~ x1 + _confounder_x1"},
        },
    )
    # Just exercise the call -- the formatting is the contract, not string equality.
    results.summary()


def test_summary_handles_empty_reports_dict():
    """`summary()` on an empty report (no models passed) should not raise."""
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()
    results = compare_models(data=df, truth=truth, models={})
    results.summary()  # must not raise


def test_summary_handles_models_with_no_applicable_metrics():
    """When no model has any applicable metric under the current weights,
    `summary()` should still print a sensible message instead of crashing."""
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=2)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()  # no confounder, no break, no HTE -> no metrics

    # Force weights to a metric that will never apply
    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1"}},
        weights={"confounding_bias": 1.0, "break_detection_lag": 1.0},
    )
    results.summary()  # must not raise; should print "(no models ...)" message
