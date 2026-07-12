"""
Metrics registration API — pluggable component metrics for StressTest.

v1 hard-coded four metrics (coef_rmse, confounding_bias, break_detection_lag,
cate_rmse-as-a-slot) directly inside StressTest.evaluate(). This module
extracts that into a registry so users can register their own metrics
without modifying spuriosity's source, and so built-in metrics are
implemented the same way a user-defined one would be -- no special-cased
internal path.

Contract
--------
A metric function has the signature:

    def metric_fn(ctx: MetricContext) -> Optional[float] | dict[str, float]:
        ...

Most metrics return a single float, or `None` if the metric does not apply
given `ctx` (e.g. no confounder was present), following the same
"absent, not zero" convention established in v1 -- StressTestReport.metrics
only contains keys for metrics that actually applied, never a placeholder
0.0 or NaN for an inapplicable one.

A metric may instead return a `dict[str, float]` for metrics that naturally
produce more than one number (e.g. `confounding_bias`, which reports both
a per-feature breakdown and an aggregate). When a metric returns a dict,
each key in that dict is namespaced under the metric's registered name as
`f"{name}:{key}"` in the final results, **except** a dict entry whose key
exactly matches the registered metric's own name, which is stored bare
(this is how `confounding_bias` produces both `confounding_bias` itself
and `confounding_bias:x1`, `confounding_bias:x2`, etc. from one registered
function). A dict-returning metric that applies to nothing should return
an empty dict `{}` (treated the same as `None`: no entries added).

`MetricContext` bundles everything a metric might plausibly need, so metric
functions have a uniform signature regardless of how much of the fit
pipeline they need to re-run (e.g. break_detection_lag needs to call
fit_fn again on sub-windows of data, not just inspect the original fit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import pandas as pd

from spuriosity.ground_truth import GroundTruth


@dataclass(frozen=True)
class MetricContext:
    """Everything a registered metric function might need to compute its
    value. Passed as a single argument so the metric_fn signature stays
    uniform across trivial metrics (e.g. coef_rmse, which only needs
    `fitted_coefficients` and `truth`) and metrics that need to re-run
    fitting (e.g. break_detection_lag, which refits on sub-windows).
    """

    truth: GroundTruth
    fitted_coefficients: dict[str, float]
    fit_result: object
    data: pd.DataFrame
    fit_fn: Callable[..., object]
    fit_kwargs: dict
    period_col: str = "period"


MetricResult = Union[float, dict[str, float], None]
MetricFn = Callable[[MetricContext], MetricResult]


class MetricRegistry:
    """A named collection of metric functions, each evaluated against a
    MetricContext and contributing one or more entries to
    StressTestReport.metrics (see the module docstring's Contract section
    for the single-float vs. dict-returning convention).

    Registries are independent objects (not process-wide global state by
    default) so tests and different StressTest instances can use isolated
    registries without interfering with each other, but
    `spuriosity.metrics.default_registry` is provided as the registry
    StressTest uses when the caller doesn't supply their own.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, MetricFn] = {}

    def register(self, name: str, fn: MetricFn) -> None:
        """Register a metric function under `name`. Re-registering an
        existing name overwrites it (allows users to override a built-in
        metric's implementation by registering under the same name)."""
        self._metrics[name] = fn

    def unregister(self, name: str) -> None:
        """Remove a registered metric by name. Raises KeyError if not
        present, so typos are caught rather than silently ignored."""
        del self._metrics[name]

    def names(self) -> list[str]:
        return list(self._metrics.keys())

    def run_all(self, ctx: MetricContext) -> dict[str, float]:
        """Evaluate every registered metric against `ctx` and return a
        dict of the applicable results. Metrics returning None (or an
        empty dict) contribute nothing -- they are omitted, not included
        as 0.0/NaN. See the module docstring for the single-float vs.
        dict-returning convention.
        """
        results: dict[str, float] = {}
        for name, fn in self._metrics.items():
            value = fn(ctx)
            if value is None:
                continue
            if isinstance(value, dict):
                for key, val in value.items():
                    result_key = name if key == name else f"{name}:{key}"
                    results[result_key] = val
            else:
                results[name] = value
        return results

    def copy(self) -> "MetricRegistry":
        """Return a shallow copy of this registry (same metric functions,
        independent registration dict) -- useful for starting from the
        built-in defaults and adding/overriding a few without mutating
        the shared `default_registry`."""
        new_registry = MetricRegistry()
        new_registry._metrics = dict(self._metrics)
        return new_registry


# ----------------------------------------------------------------------
# Built-in metrics (ported from v1's hard-coded StressTest.evaluate logic)
# ----------------------------------------------------------------------


def _coef_rmse(ctx: MetricContext) -> Optional[float]:
    """RMSE between fitted and true coefficients, over keys present in
    both. Returns None (not 0.0) if there's no overlap, or if
    truth.true_coefficients is empty."""
    import numpy as np

    true = ctx.truth.true_coefficients
    fitted = ctx.fitted_coefficients
    if not true:
        return None
    shared_keys = [k for k in true if k in fitted]
    if not shared_keys:
        return None
    errors = np.array([fitted[k] - true[k] for k in shared_keys])
    return float(np.sqrt(np.mean(errors**2)))


def _confounding_bias(ctx: MetricContext) -> Optional[dict[str, float]]:
    """Per-confounded-feature |fitted - true| coefficient bias, plus an
    aggregate mean under the bare 'confounding_bias' key. Returns None if
    truth.confounding_strength is empty or no confounded feature has both
    a true and fitted coefficient available."""
    import numpy as np

    if not ctx.truth.confounding_strength:
        return None

    result: dict[str, float] = {}
    for feature in ctx.truth.confounding_strength:
        true_val = ctx.truth.true_coefficients.get(feature)
        fitted_val = ctx.fitted_coefficients.get(feature)
        if true_val is not None and fitted_val is not None:
            result[feature] = abs(fitted_val - true_val)

    if not result:
        return None

    result["confounding_bias"] = float(np.mean(list(result.values())))
    return result


def _break_detection_lag(ctx: MetricContext) -> Optional[float]:
    """Estimate how many periods after the true break the model's fitted
    coefficient actually shifts, via rolling-window refits.

    For each structural break with kind="coefficient_shift" (the only kind
    with a well-defined "which coefficient changed" target), refits the
    same model spec on each period window and finds the first period where
    the fitted coefficient crosses halfway between the true pre-break and
    post-break values. Returns the signed lag (detected_period -
    true_break_period); positive means detection lagged the true break,
    negative means it was detected early (e.g. due to noise), 0 means
    exact detection at the resolution of the window.

    Returns None if there's no coefficient_shift break, no formula=
    fit_kwarg to reuse across windows, or the target coefficient never
    crosses the halfway threshold in any window.
    """
    truth = ctx.truth
    coef_shift_breaks = [b for b in truth.break_points if b.kind == "coefficient_shift"]
    if not coef_shift_breaks:
        return None
    if ctx.fit_kwargs.get("formula") is None:
        return None

    brk = coef_shift_breaks[0]
    data = ctx.data
    period_col = ctx.period_col
    fit_fn = ctx.fit_fn
    fit_kwargs = ctx.fit_kwargs

    periods = sorted(data[period_col].unique())
    if len(periods) < 2:
        return None

    pre_period_data = data[data[period_col] < brk.period]
    post_period_data = data[data[period_col] >= brk.period]
    if pre_period_data.empty or post_period_data.empty:
        return None

    pre_fit = _extract_coefficients_for_metrics(fit_fn(pre_period_data, **fit_kwargs))
    post_fit = _extract_coefficients_for_metrics(fit_fn(post_period_data, **fit_kwargs))

    changed_key = None
    max_delta = 0.0
    for key in pre_fit:
        if key in post_fit:
            delta = abs(post_fit[key] - pre_fit[key])
            if delta > max_delta:
                max_delta = delta
                changed_key = key

    if changed_key is None or max_delta < 1e-6:
        return None

    pre_val = pre_fit[changed_key]
    post_val = post_fit[changed_key]
    halfway = (pre_val + post_val) / 2.0

    window_size = 1
    detected_period = None
    for p in periods:
        window = data[(data[period_col] >= p) & (data[period_col] < p + window_size)]
        if window.empty:
            continue
        window_fit = _extract_coefficients_for_metrics(fit_fn(window, **fit_kwargs))
        val = window_fit.get(changed_key)
        if val is None:
            continue
        crossed = (val >= halfway) if post_val > pre_val else (val <= halfway)
        if crossed:
            detected_period = p
            break

    if detected_period is None:
        return None

    return float(detected_period - brk.period)


def _extract_coefficients_for_metrics(fit_result: object) -> dict[str, float]:
    """Local copy of the coefficient-extraction helper (also used by
    StressTest.evaluate itself); duplicated here rather than imported from
    stress_test.py to avoid a circular import between the two modules."""
    if isinstance(fit_result, dict):
        return dict(fit_result)
    coefficients = getattr(fit_result, "coefficients", None)
    if coefficients is not None:
        return dict(coefficients)
    return {}


def _cate_rmse(ctx: MetricContext) -> Optional[float]:
    """Placeholder slot for CATE recovery error, carried over from v1
    (see stress_test.py's original module docstring): full implementation
    is deferred pending an established convention for how a fit_fn exposes
    row-level treatment effect predictions. Always returns None for now,
    so it never appears in StressTestReport.metrics, matching v1's
    behavior exactly (cate_rmse was a composite-weight slot but never
    actually computed in v1 either).
    """
    return None


default_registry = MetricRegistry()
default_registry.register("coef_rmse", _coef_rmse)
default_registry.register("confounding_bias", _confounding_bias)
default_registry.register("break_detection_lag", _break_detection_lag)
default_registry.register("cate_rmse", _cate_rmse)
