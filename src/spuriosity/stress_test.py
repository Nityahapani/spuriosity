"""
StressTest — evaluate a single model/estimator against a GroundTruth.
compare_models — run several models against the same DGP and produce a
ranked benchmark report.

StressTest is function-based (fit_fn/predict_fn) for maximum flexibility;
spuriosity.reference provides common fits out of the box.
"""

from __future__ import annotations

from typing import Callable

from spuriosity.ground_truth import GroundTruth


class StressTestReport:
    """Result of evaluating one model against ground truth."""

    def summary(self) -> None:
        raise NotImplementedError


class StressTest:
    def __init__(self, truth: GroundTruth) -> None:
        self.truth = truth

    def evaluate(self, fit_fn: Callable, predict_fn: Callable, data) -> StressTestReport:
        raise NotImplementedError


class ComparisonReport:
    """Result of compare_models — ranked table plus per-model, per-metric
    breakdown. Composite ranking is transparent and user-overridable; see
    docs/design_spec.md."""

    def ranked_table(self, by: str = "default_composite"):
        raise NotImplementedError


def compare_models(
    data,
    truth: GroundTruth,
    models: dict[str, tuple[Callable, Callable]],
    weights: dict[str, float] | None = None,
) -> ComparisonReport:
    """Run multiple models against the same DGP/ground truth.

    `weights` overrides the default composite score weights (default: 1.0
    for each applicable component metric). Individual metrics are always
    exposed regardless of composite weighting — see ComparisonReport.
    """
    raise NotImplementedError
