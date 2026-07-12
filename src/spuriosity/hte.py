"""
HTE — single-dimension heterogeneous treatment effect specification.

Unlike the pathologies in `pathologies.py`, HTE is not a failure mode being
injected -- it's a legitimate DGP feature (the treatment effect genuinely
varies by a covariate) that `spuriosity` supports because recovering
heterogeneous effects correctly is itself a common thing to stress-test
(e.g. checking whether a DoubleML/causal-forest style estimator recovers
the true CATE function, not just the ATE).

v1 supports a single modifier dimension only; the resulting `true_cate` is
a clean 1D callable, easy to reason about and plot. Multi-dimensional
modifiers are deferred to v1.1 (see docs/design_spec.md).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class HTE:
    """Specifies that `treatment`'s effect on the outcome varies with
    `modifier` according to `formula`, a `pandas.eval`-evaluated expression
    in terms of `modifier` (e.g. ``"3 + 1.5*x1"``).

    When an HTE is added for a given treatment, it *replaces* that
    treatment's contribution to the outcome mean entirely -- any
    coefficient supplied for the treatment column in
    `PanelGenerator.set_outcome(coefficients=...)` is ignored for the
    purposes of the treatment's effect (a warning is emitted if a nonzero
    coefficient was supplied, since this likely indicates the two features
    are being used together by mistake). The outcome becomes:

        outcome_mean = <rest of formula, excluding treatment term>
                       + cate_fn(modifier) * treatment

    `GroundTruth.true_cate` exposes `cate_fn` directly as a plain Python
    callable taking the modifier value(s) and returning the treatment
    effect at that point, e.g. ``truth.true_cate(1.0)``.
    """

    def __init__(self, treatment: str, modifier: str, formula: str) -> None:
        self.treatment = treatment
        self.modifier = modifier
        self.formula = formula

    def cate_fn(self) -> Callable[[float], float]:
        """Return a plain Python callable `f(modifier_value) -> effect`,
        suitable for `GroundTruth.true_cate` and for plotting."""
        formula = self.formula
        modifier = self.modifier

        def _fn(modifier_value: float) -> float:
            result = pd.eval(
                formula,
                local_dict={modifier: modifier_value},
                global_dict={},
                engine="python",
            )
            return float(result)

        return _fn

    def evaluate_on_column(self, modifier_values: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of the CATE formula across an array of
        modifier values, used internally by PanelGenerator to compute each
        row's individual treatment effect during generation."""
        try:
            result = pd.eval(
                self.formula,
                local_dict={self.modifier: pd.Series(modifier_values)},
                global_dict={},
                engine="python",
            )
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate HTE formula {self.formula!r} for modifier "
                f"{self.modifier!r}: {e}"
            ) from e

        arr = np.asarray(result, dtype=float)
        if arr.shape == ():
            arr = np.full(modifier_values.shape, float(arr))
        return arr
