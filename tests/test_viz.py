"""Tests for spuriosity.viz.plot_recovery_report.

Uses matplotlib's non-interactive 'Agg' backend (set in conftest.py) so
these run headlessly in CI/sandbox environments without a display.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spuriosity import PanelGenerator, reference  # noqa: E402
from spuriosity.stress_test import (  # noqa: E402
    ComparisonReport,
    StressTest,
    StressTestReport,
    compare_models,
)
from spuriosity.viz import plot_recovery_report  # noqa: E402


def _basic_stress_test_report(n_entities: int = 5000, seed: int = 42) -> StressTestReport:
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(
        formula="x1 + treat", coefficients={"x1": 2.0, "treat": 3.0, "Intercept": 1.0}, noise_std=0.5
    )
    df, truth = gen.generate()
    test = StressTest(truth)
    return test.evaluate(
        fit_fn=reference.ols_fit,
        predict_fn=reference.ols_predict,
        data=df,
        fit_kwargs={"formula": "y ~ x1 + treat"},
        model_name="OLS",
    )


def _hte_stress_test_report(n_entities: int = 1000, seed: int = 42) -> StressTestReport:
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0, "Intercept": 0.0}, noise_std=0.1
    )
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    df, truth = gen.generate()
    test = StressTest(truth)
    return test.evaluate(
        fit_fn=reference.ols_fit,
        predict_fn=reference.ols_predict,
        data=df,
        fit_kwargs={"formula": "y ~ x1 + treat"},
        model_name="OLS_HTE",
    )


def _basic_comparison_report(n_entities: int = 5000, seed: int = 42) -> ComparisonReport:
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_confounder(feature="x1", outcome="y", strength=0.5, observed=True)
    df, truth = gen.generate()
    return compare_models(
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


def test_plot_stress_test_report_coefficients_only_returns_figure():
    report = _basic_stress_test_report()
    fig = plot_recovery_report(report)
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_plot_stress_test_report_with_cate_range_creates_two_panels():
    report = _hte_stress_test_report()
    fig = plot_recovery_report(report, cate_range=(-3, 3))
    assert len(fig.axes) == 2


def test_plot_stress_test_report_without_cate_range_creates_one_panel():
    report = _hte_stress_test_report()
    fig = plot_recovery_report(report)  # no cate_range supplied
    assert len(fig.axes) == 1


def test_plot_comparison_report_returns_figure():
    results = _basic_comparison_report()
    fig = plot_recovery_report(results)
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_plot_comparison_report_has_one_bar_per_model():
    results = _basic_comparison_report()
    fig = plot_recovery_report(results)
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 2  # two models in this report


def test_plot_rejects_wrong_type():
    with pytest.raises(TypeError, match="expects a StressTestReport or ComparisonReport"):
        plot_recovery_report("not a report")  # type: ignore[arg-type]


def test_plot_empty_stress_test_report_raises():
    empty = StressTestReport(model_name="empty", metrics={}, fitted_coefficients={}, true_coefficients={})
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_recovery_report(empty)


def test_plot_empty_comparison_report_raises():
    empty = ComparisonReport(reports={}, weights={})
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_recovery_report(empty)


def test_plot_saves_to_file(tmp_path):
    report = _basic_stress_test_report()
    fig = plot_recovery_report(report)
    out_path = tmp_path / "test_plot.png"
    fig.savefig(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_save_path_kwarg_writes_file_and_returns_figure(tmp_path):
    """`save_path=` is a one-shot convenience: save the figure and return it.
    The Figure object is always returned so the caller can still show / save
    it elsewhere if they want."""
    report = _basic_stress_test_report()
    out_path = tmp_path / "via_kwarg.png"
    fig = plot_recovery_report(report, save_path=str(out_path))
    assert fig is not None
    assert hasattr(fig, "savefig")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_save_path_kwarg_works_for_comparison_report(tmp_path):
    """`save_path=` also works for ComparisonReport, not just StressTestReport."""
    cmp = _basic_comparison_report()
    out_path = tmp_path / "comparison.png"
    fig = plot_recovery_report(cmp, save_path=str(out_path))
    assert fig is not None
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_save_path_none_does_not_write(tmp_path):
    """When `save_path=None` (the default), no file is written and the
    Figure is still returned."""
    report = _basic_stress_test_report()
    # tmp_path is empty before, should still be empty after
    fig = plot_recovery_report(report)
    assert fig is not None
    assert list(tmp_path.iterdir()) == []  # no files written
