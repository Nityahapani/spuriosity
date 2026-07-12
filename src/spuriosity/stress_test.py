"""
StressTest — evaluate a single model/estimator against a GroundTruth.
compare_models — run several models against the same DGP and produce a
ranked benchmark report.

StressTest is function-based (fit_fn/predict_fn) for maximum flexibility;
spuriosity.reference provides common fits out of the box (see
spuriosity.reference.FitResult for the expected fit_fn return contract).

Component metrics (each computed only when the relevant ground-truth field
is present):

  - coef_rmse: RMSE between fit.coefficients and truth.true_coefficients,
    over keys present in truth.true_coefficients. Always computed if
    truth.true_coefficients is non-empty and the fit_fn returns a
    FitResult-like object with .coefficients.
  - cate_rmse: only computed if truth.true_cate is set. Requires the model
    to expose a per-row treatment effect; see StressTest.evaluate's
    `treatment_col`/`modifier_col` parameters.
  - confounding_bias: only computed if truth.confounding_strength is set.
    For each confounded feature, the absolute difference between the
    fitted coefficient on that feature and its true coefficient.
  - break_detection_lag: only computed if truth.break_points is set.
    Estimated via rolling-window refits of the same model spec across
    periods, detecting where the target coefficient crosses halfway
    between its pre- and post-break true values, compared against the
    true break period.

All component metrics are always exposed on StressTestReport, regardless
of the composite weighting used in compare_models. See docs/design_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from spuriosity.ground_truth import GroundTruth
from spuriosity.metrics import MetricContext, MetricRegistry, default_registry

_DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "coef_rmse": 1.0,
    "cate_rmse": 1.0,
    "confounding_bias": 1.0,
    "break_detection_lag": 1.0,
}


@dataclass
class StressTestReport:
    """Result of evaluating one model against ground truth.

    `metrics` holds every component metric that was applicable given the
    GroundTruth's populated fields (see module docstring). Metrics that
    don't apply (e.g. cate_rmse when there was no HTE) are simply absent
    from the dict, not set to a placeholder value like 0 or NaN.
    """

    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    fitted_coefficients: dict[str, float] = field(default_factory=dict)
    true_coefficients: dict[str, float] = field(default_factory=dict)
    true_cate: Optional[Callable[[float], float]] = None

    def summary(self) -> None:
        """Print a human-readable summary of this report."""
        print(f"StressTestReport: {self.model_name}")
        if self.true_coefficients:
            print("  Coefficient recovery:")
            for key, true_val in self.true_coefficients.items():
                fitted_val = self.fitted_coefficients.get(key)
                if fitted_val is None:
                    print(f"    {key}: true={true_val:.4f}, fitted=<missing>")
                else:
                    print(f"    {key}: true={true_val:.4f}, fitted={fitted_val:.4f}, "
                          f"error={fitted_val - true_val:+.4f}")
        if self.metrics:
            print("  Metrics:")
            for key, val in self.metrics.items():
                print(f"    {key}: {val:.4f}")


class StressTest:
    def __init__(self, truth: GroundTruth, metric_registry: Optional[MetricRegistry] = None) -> None:
        """`metric_registry` defaults to `spuriosity.metrics.default_registry`
        (the four built-in metrics: coef_rmse, confounding_bias,
        break_detection_lag, cate_rmse). Pass a custom registry -- e.g.
        `default_registry.copy()` with additional metrics registered, or a
        fresh `MetricRegistry()` for a from-scratch metric set -- to
        customize which metrics get computed.
        """
        self.truth = truth
        self.metric_registry = metric_registry if metric_registry is not None else default_registry

    def evaluate(
        self,
        fit_fn: Callable[..., object],
        predict_fn: Callable[..., np.ndarray],
        data: pd.DataFrame,
        fit_kwargs: Optional[dict] = None,
        model_name: str = "model",
        period_col: str = "period",
    ) -> StressTestReport:
        """Fit `fit_fn` on `data` and score it against `self.truth` using
        every metric registered in `self.metric_registry`.

        `fit_kwargs` are passed through to `fit_fn(data, **fit_kwargs)`.
        `predict_fn` is retained on the signature for API symmetry with
        `spuriosity.reference` and for future metrics that need row-level
        predictions; no built-in metric currently calls it directly (they
        re-run `fit_fn` on sub-windows instead, e.g. break_detection_lag).
        """
        fit_kwargs = fit_kwargs or {}
        fit_result = fit_fn(data, **fit_kwargs)
        fitted_coefficients = _extract_coefficients(fit_result)

        ctx = MetricContext(
            truth=self.truth,
            fitted_coefficients=fitted_coefficients,
            fit_result=fit_result,
            data=data,
            fit_fn=fit_fn,
            fit_kwargs=fit_kwargs,
            period_col=period_col,
        )
        metrics = self.metric_registry.run_all(ctx)

        return StressTestReport(
            model_name=model_name,
            metrics=metrics,
            fitted_coefficients=fitted_coefficients,
            true_coefficients=dict(self.truth.true_coefficients),
            true_cate=self.truth.true_cate,
        )


def _extract_coefficients(fit_result: object) -> dict[str, float]:
    """Extract a coefficients dict from a fit_fn's return value. Supports
    both spuriosity.reference.FitResult (via .coefficients) and a plain
    dict returned directly by a user-supplied fit_fn."""
    if isinstance(fit_result, dict):
        return dict(fit_result)
    coefficients = getattr(fit_result, "coefficients", None)
    if coefficients is not None:
        return dict(coefficients)
    return {}


# ----------------------------------------------------------------------
# compare_models
# ----------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Result of compare_models: per-model StressTestReports plus a
    transparent, user-overridable composite ranking.

    Individual component metrics are always available via `.reports`
    regardless of composite weighting -- the composite is a convenience
    view, not the only way to inspect results.
    """

    reports: dict[str, StressTestReport]
    weights: dict[str, float]

    def ranked_table(self, by: str = "default_composite") -> pd.DataFrame:
        """Return a DataFrame of models ranked by `by` (ascending -- lower
        error is better for every metric spuriosity currently computes).

        `by="default_composite"` ranks by the weighted sum of applicable
        component metrics (see `self.weights`). Any other value ranks by
        that single metric name directly (e.g. `by="coef_rmse"`); models
        for which that metric wasn't computed are excluded from the
        ranking (not silently scored as 0 or infinity), with a note in
        the returned DataFrame's attrs.
        """
        rows = []
        excluded = []
        for model_name, report in self.reports.items():
            if by == "default_composite":
                score = self._composite_score(report)
                if score is None:
                    excluded.append(model_name)
                    continue
                row = {"model": model_name, "default_composite": score}
                row.update(report.metrics)
            else:
                if by not in report.metrics:
                    excluded.append(model_name)
                    continue
                row = {"model": model_name, by: report.metrics[by]}
                row.update(report.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=by).reset_index(drop=True)
        df.attrs["excluded_models"] = excluded
        return df

    def summary(self) -> None:
        """Print a human-readable summary of this comparison report.

        Mirrors `StressTestReport.summary()` so the two report types have a
        consistent ergonomics: per-model coefficient recovery + metrics
        first, then the ranked table (best-to-worst by default_composite),
        then any models excluded from the default-composite ranking because
        none of their metrics were applicable under the current weights.
        """
        print(f"ComparisonReport: {len(self.reports)} models")
        if not self.reports:
            return

        print("  Composite weights: " + ", ".join(
            f"{k}={v}" for k, v in sorted(self.weights.items())
        ))

        print("\n  Per-model results:")
        for model_name, report in self.reports.items():
            print(f"    [{model_name}]")
            if report.true_coefficients:
                print("      Coefficient recovery:")
                for key, true_val in report.true_coefficients.items():
                    fitted_val = report.fitted_coefficients.get(key)
                    if fitted_val is None:
                        print(f"        {key}: true={true_val:.4f}, fitted=<missing>")
                    else:
                        print(
                            f"        {key}: true={true_val:.4f}, "
                            f"fitted={fitted_val:.4f}, error={fitted_val - true_val:+.4f}"
                        )
            if report.metrics:
                print("      Metrics:")
                for key, val in report.metrics.items():
                    print(f"        {key}: {val:.4f}")
            elif not report.true_coefficients:
                print("      (no applicable metrics for this model)")

        print("\n  Ranked table (best-to-worst by default_composite):")
        try:
            ranked = self.ranked_table()
        except Exception as e:
            print(f"    (could not rank: {type(e).__name__}: {e})")
            return
        if ranked.empty:
            print("    (no models had any applicable metrics under the current weights)")
        else:
            # Truncate long model names + cap float precision for terminal readability
            with pd.option_context("display.max_colwidth", 30, "display.precision", 4):
                print(ranked.to_string(index=False))
        excluded = ranked.attrs.get("excluded_models", []) if not ranked.empty else []
        if excluded:
            print(f"  Excluded from default_composite (no applicable metrics): {excluded}")

    def _composite_score(self, report: StressTestReport) -> Optional[float]:
        applicable = {k: v for k, v in report.metrics.items() if k in self.weights}
        if not applicable:
            return None
        return float(sum(self.weights[k] * v for k, v in applicable.items()))


def compare_models(
    data: pd.DataFrame,
    truth: GroundTruth,
    models: dict[str, tuple[Callable[..., object], Callable[..., np.ndarray]]],
    weights: Optional[dict[str, float]] = None,
    fit_kwargs_per_model: Optional[dict[str, dict]] = None,
    metric_registry: Optional[MetricRegistry] = None,
) -> ComparisonReport:
    """Run multiple models against the same DGP/ground truth.

    `models` maps a display name to a `(fit_fn, predict_fn)` pair (matching
    the `spuriosity.reference` function signatures).

    `weights` overrides the default composite score weights (default: 1.0
    for each applicable component metric -- see module-level
    `_DEFAULT_COMPOSITE_WEIGHTS`). Individual metrics are always exposed
    regardless of composite weighting; see `ComparisonReport`.

    `fit_kwargs_per_model` optionally supplies per-model `fit_kwargs` (e.g.
    a different `formula=` string per model), keyed by the same display
    names used in `models`.

    `metric_registry` is passed through to the underlying `StressTest` for
    every model (the same registry is used for all models being compared,
    so their metrics remain directly comparable). Defaults to
    `spuriosity.metrics.default_registry`.
    """
    effective_weights = dict(_DEFAULT_COMPOSITE_WEIGHTS)
    if weights is not None:
        effective_weights.update(weights)

    fit_kwargs_per_model = fit_kwargs_per_model or {}

    stress_test = StressTest(truth, metric_registry=metric_registry)
    reports = {}
    for model_name, (fit_fn, _predict_fn) in models.items():
        kwargs = fit_kwargs_per_model.get(model_name, {})
        reports[model_name] = stress_test.evaluate(
            fit_fn=fit_fn, predict_fn=_predict_fn, data=data, fit_kwargs=kwargs, model_name=model_name
        )

    return ComparisonReport(reports=reports, weights=effective_weights)
