"""
Visualization for stress test and comparison reports: coefficient recovery
vs. ground truth, pre/post structural-break performance, CATE estimates vs.
true CATE, and multi-model comparison rankings.

Requires the `viz` optional dependency group (matplotlib).
"""

from __future__ import annotations

from typing import Union

from spuriosity.stress_test import StressTestReport, ComparisonReport


def plot_recovery_report(report: Union[StressTestReport, ComparisonReport]):
    raise NotImplementedError
