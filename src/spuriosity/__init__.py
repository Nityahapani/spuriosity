"""
spuriosity — synthetic panel data with known ground truth and injectable
econometric pathologies, for stress-testing ML pipelines and benchmarking
causal inference methods.

See docs/design_spec.md in the repository for the full design rationale.
"""

__version__ = "0.1.0"

from spuriosity import reference
from spuriosity.generator import PanelGenerator
from spuriosity.ground_truth import (
    BreakInfo,
    EndogeneityInfo,
    GroundTruth,
    HeteroskedasticityInfo,
    MeasurementErrorInfo,
    MulticollinearityInfo,
    SelectionInfo,
)
from spuriosity.hte import HTE
from spuriosity.metrics import MetricContext, MetricRegistry, default_registry
from spuriosity.pathologies import (
    Confounder,
    Endogeneity,
    Heteroskedasticity,
    MeasurementError,
    Multicollinearity,
    Pathology,
    SelectionBias,
    StructuralBreak,
)
from spuriosity.stress_test import ComparisonReport, StressTest, StressTestReport, compare_models
from spuriosity.viz import plot_recovery_report

__all__ = [
    "PanelGenerator",
    "GroundTruth",
    "BreakInfo",
    "SelectionInfo",
    "HeteroskedasticityInfo",
    "MulticollinearityInfo",
    "MeasurementErrorInfo",
    "EndogeneityInfo",
    "HTE",
    "reference",
    "StressTest",
    "StressTestReport",
    "compare_models",
    "ComparisonReport",
    "plot_recovery_report",
    "MetricContext",
    "MetricRegistry",
    "default_registry",
    "Pathology",
    "StructuralBreak",
    "Confounder",
    "SelectionBias",
    "Heteroskedasticity",
    "Multicollinearity",
    "MeasurementError",
    "Endogeneity",
]
