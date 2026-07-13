"""
HTE — heterogeneous treatment effect specification, single- or
multi-dimensional.

Unlike the pathologies in `pathologies.py`, HTE is not a failure mode being
injected -- it's a legitimate DGP feature (the treatment effect genuinely
varies by one or more covariates) that `spuriosity` supports because
recovering heterogeneous effects correctly is itself a common thing to
stress-test (e.g. checking whether a DoubleML/causal-forest style
estimator recovers the true CATE function, not just the ATE).

v1 shipped single-dimension modifiers only. v2 extends `modifier` to accept
either a single column name (str) or a list of column names, while keeping
the single-dimension case's `cate_fn()` signature exactly
`Callable[[float], float]` -- unchanged from v1 -- so no existing code that
calls `truth.true_cate(x)` breaks. The multi-dimension case's `cate_fn()`
instead returns `Callable[..., float]` accepting keyword arguments named
after each modifier (e.g. `truth.true_cate(x1=1.0, x2=2.0)`), since a
positional-only multi-arg signature would be ambiguous about argument
order without the caller consulting `HTE.modifier`/`GroundTruth`-level
metadata every time.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd


class HTE:
    """Specifies that `treatment`'s effect on the outcome varies with one
    or more `modifier` covariates according to `formula`, a
    `pandas.eval`-evaluated expression in terms of the modifier name(s)
    (e.g. ``"3 + 1.5*x1"`` for a single modifier, or
    ``"3 + 1.5*x1 - 0.5*x2"`` for two).

    `modifier` accepts either a single column name (str) -- the v1
    single-dimension case -- or a list of column names for a
    multi-dimensional CATE surface. Internally these are handled
    uniformly; the only user-visible difference is the shape of the
    callable returned by `cate_fn()`:

    - Single modifier (`modifier: str`, or a length-1 list): `cate_fn()`
      returns `Callable[[float], float]`, called positionally as
      `f(x)` -- exactly the v1 signature, unchanged.
    - Multiple modifiers (`modifier: list[str]` with 2+ entries):
      `cate_fn()` returns a callable that must be called with keyword
      arguments matching each modifier's name, e.g. `f(x1=1.0, x2=2.0)`.
      This avoids positional-argument-order ambiguity.

    When an HTE is added for a given treatment, it *replaces* that
    treatment's contribution to the outcome mean entirely -- any
    coefficient supplied for the treatment column in
    `PanelGenerator.set_outcome(coefficients=...)` is ignored for the
    purposes of the treatment's effect (a warning is emitted if a nonzero
    coefficient was supplied, since this likely indicates the two features
    are being used together by mistake). The outcome becomes:

        outcome_mean = <rest of formula, excluding treatment term>
                       + cate_fn(modifier(s)) * treatment

    `GroundTruth.true_cate` exposes `cate_fn()`'s return value directly.
    """

    def __init__(self, treatment: str, modifier: Union[str, list[str]], formula: str) -> None:
        self.treatment = treatment
        self.modifiers: list[str] = [modifier] if isinstance(modifier, str) else list(modifier)
        if not self.modifiers:
            raise ValueError("modifier must be a non-empty string or list of strings")
        self.formula = formula

    @property
    def is_multi_dim(self) -> bool:
        return len(self.modifiers) > 1

    @property
    def modifier(self) -> str:
        """The single modifier name, for backward compatibility with v1
        code that reads `hte.modifier` expecting a single string. Raises
        if this HTE is multi-dimensional -- use `.modifiers` instead in
        that case."""
        if self.is_multi_dim:
            raise AttributeError(
                "This HTE has multiple modifiers; use `.modifiers` (a list) instead of "
                "the single-dimension `.modifier` accessor."
            )
        return self.modifiers[0]

    def cate_fn(self) -> Union[Callable[[float], float], Callable[..., float]]:
        """Return a plain Python callable for the true treatment effect.

        Single modifier: `Callable[[float], float]`, called positionally
        (`f(x)`) -- identical to v1's signature.

        Multiple modifiers: a callable that must be called with keyword
        arguments matching each modifier's name (`f(x1=1.0, x2=2.0)`).
        """
        formula = self.formula
        modifiers = self.modifiers

        if not self.is_multi_dim:
            single_modifier = modifiers[0]

            def _single_fn(modifier_value: float) -> float:
                result = pd.eval(
                    formula,
                    local_dict={single_modifier: modifier_value},
                    global_dict={},
                    engine="python",
                )
                return float(result)

            return _single_fn

        def _multi_fn(**kwargs: float) -> float:
            missing = [m for m in modifiers if m not in kwargs]
            if missing:
                raise TypeError(
                    f"true_cate() missing required keyword argument(s) for modifier(s): "
                    f"{missing}. This HTE is multi-dimensional (modifiers={modifiers}); "
                    f"call it as true_cate({', '.join(f'{m}=...' for m in modifiers)})."
                )
            extra = [k for k in kwargs if k not in modifiers]
            if extra:
                raise TypeError(
                    f"true_cate() got unexpected keyword argument(s): {extra}. "
                    f"This HTE's modifiers are: {modifiers}."
                )
            local_dict = {m: kwargs[m] for m in modifiers}
            result = pd.eval(formula, local_dict=local_dict, global_dict={}, engine="python")
            return float(result)

        return _multi_fn

    def evaluate_on_columns(self, modifier_values: dict[str, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of the CATE formula across arrays of
        modifier values (one array per modifier, keyed by modifier name),
        used internally by PanelGenerator to compute each row's individual
        treatment effect during generation. Works uniformly for both the
        single- and multi-dimensional case.
        """
        missing = [m for m in self.modifiers if m not in modifier_values]
        if missing:
            raise ValueError(
                f"evaluate_on_columns missing arrays for modifier(s): {missing}"
            )
        local_dict = {m: pd.Series(modifier_values[m]) for m in self.modifiers}
        try:
            result = pd.eval(self.formula, local_dict=local_dict, global_dict={}, engine="python")
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate HTE formula {self.formula!r} for modifier(s) "
                f"{self.modifiers!r}: {e}"
            ) from e

        # shape reference from any one of the modifier arrays
        ref_shape = next(iter(modifier_values.values())).shape
        arr = np.asarray(result, dtype=float)
        if arr.shape == ():
            arr = np.full(ref_shape, float(arr))
        return arr

    def evaluate_on_column(self, modifier_values: np.ndarray) -> np.ndarray:
        """Backward-compatible single-modifier vectorized evaluation
        (identical to v1's `evaluate_on_column`). Only valid when this HTE
        has exactly one modifier; raises otherwise, directing callers to
        `evaluate_on_columns` (plural) for the multi-dimensional case."""
        if self.is_multi_dim:
            raise AttributeError(
                "This HTE has multiple modifiers; use evaluate_on_columns({modifier_name: "
                "array, ...}) instead of the single-modifier evaluate_on_column()."
            )
        return self.evaluate_on_columns({self.modifiers[0]: modifier_values})

