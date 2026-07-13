"""
Visualization for stress test and comparison reports: coefficient recovery
vs. ground truth, true CATE curves, and multi-model comparison rankings.

Requires the `viz` optional dependency group (matplotlib):
``pip install spuriosity[viz]``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from spuriosity.stress_test import ComparisonReport, StressTestReport


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_recovery_report requires the optional 'viz' dependency (matplotlib). "
            "Install it with: pip install spuriosity[viz]"
        ) from e
    return plt


def plot_recovery_report(
    report: Union[StressTestReport, ComparisonReport],
    cate_range: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
):
    """Visualize a StressTestReport or ComparisonReport.

    For a `StressTestReport`: a bar chart of true vs. fitted coefficients
    for every shared key, plus (if the model's report has a `true_cate`
    curve available via the caller passing `cate_range`) a plotted true
    CATE curve. Coefficient recovery is always plotted if
    `report.true_coefficients` is non-empty.

    For a `ComparisonReport`: a horizontal bar chart of each model's
    `default_composite` score, sorted best-to-worst (lowest error first),
    annotated with each model's individual component metrics.

    The `Figure` object is always returned so the caller can save / show
    / customize it themselves. For the common one-shot case, pass
    `save_path=` and the figure will be saved there (after `tight_layout`)
    before being returned; nothing is shown or closed automatically.

    Returns the created matplotlib Figure.
    """
    plt = _require_matplotlib()

    if isinstance(report, StressTestReport):
        fig = _plot_stress_test_report(plt, report, cate_range)
    elif isinstance(report, ComparisonReport):
        fig = _plot_comparison_report(plt, report)
    else:
        raise TypeError(
            f"plot_recovery_report expects a StressTestReport or ComparisonReport, "
            f"got {type(report).__name__}"
        )

    if save_path is not None:
        # tight_layout was already applied by the per-report helper; just save.
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def _plot_stress_test_report(plt, report: StressTestReport, cate_range: Optional[tuple[float, float]]):
    has_coefficients = bool(report.true_coefficients)
    has_cate = report.true_cate is not None and cate_range is not None

    n_panels = int(has_coefficients) + int(has_cate)
    if n_panels == 0:
        raise ValueError(
            "Nothing to plot: report has no true_coefficients and no true_cate "
            "(or cate_range was not supplied)."
        )

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if has_coefficients:
        ax = axes[panel_idx]
        keys = list(report.true_coefficients.keys())
        true_vals = [report.true_coefficients[k] for k in keys]
        fitted_vals = [report.fitted_coefficients.get(k, np.nan) for k in keys]

        x = np.arange(len(keys))
        width = 0.35
        ax.bar(x - width / 2, true_vals, width, label="True")
        ax.bar(x + width / 2, fitted_vals, width, label="Fitted")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylabel("Coefficient value")
        ax.set_title(f"{report.model_name}: coefficient recovery")
        ax.legend()
        ax.axhline(0, color="gray", linewidth=0.5)
        panel_idx += 1

    if has_cate:
        ax = axes[panel_idx]
        assert cate_range is not None
        assert report.true_cate is not None
        xs = np.linspace(cate_range[0], cate_range[1], 200)
        try:
            ys = [report.true_cate(x) for x in xs]
        except TypeError as e:
            raise ValueError(
                "plot_recovery_report's CATE panel only supports single-dimension "
                "true_cate callables (called positionally as f(x)). This report's "
                "true_cate appears to require keyword arguments (a multi-dimensional "
                "HTE) and is not yet plottable -- multi-dimensional CATE surface "
                "plotting is not implemented."
            ) from e
        ax.plot(xs, ys, label="True CATE", color="black")
        ax.set_xlabel("Modifier value")
        ax.set_ylabel("Treatment effect")
        ax.set_title(f"{report.model_name}: true CATE")
        ax.legend()

    fig.tight_layout()
    return fig


def _plot_comparison_report(plt, report: ComparisonReport):
    table = report.ranked_table(by="default_composite")
    if table.empty:
        raise ValueError(
            "Nothing to plot: no model in this ComparisonReport has an applicable "
            "default_composite score."
        )

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(table) + 2))
    y_pos = np.arange(len(table))
    # Reverse so the best (lowest-error, first-ranked) model appears at the top.
    ax.barh(y_pos, table["default_composite"][::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(table["model"][::-1])
    ax.set_xlabel("Default composite score (lower is better)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    return fig
