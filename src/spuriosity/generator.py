"""
PanelGenerator — the main builder for synthetic panel datasets.

Data shape: long-format panel, one row per (entity, period). Columns are
entity_id, period, each declared variable (in declaration order), the
treatment indicator (if any), and the outcome.

Outcome specification uses one of two paths:
  - `formula=`: a patsy formula string describing the right-hand-side design
    matrix (e.g. "x1 + x2 + treat"). Patsy builds the design matrix (handling
    the intercept, categoricals, interactions, etc.); `coefficients=` then
    supplies the true coefficient for each resulting design-matrix column.
    This is the statistically correct use of patsy: formulas describe
    structure, not magnitudes -- coefficients are supplied separately.
  - `fn=`: a Python callable taking the declared variables (and treatment,
    if present) as keyword arguments and returning the noiseless outcome
    directly, for cases too nonlinear/complex for a linear design matrix.

See docs/design_spec.md for the full API design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import patsy

from spuriosity._rng import RNGManager
from spuriosity.ground_truth import GroundTruth

_SUPPORTED_DISTS = ("normal", "uniform")


@dataclass
class _VariableSpec:
    name: str
    dist: str
    params: dict


@dataclass
class _TreatmentSpec:
    name: str
    assignment: Literal["random"]
    start_period: int
    propensity: float


@dataclass
class _OutcomeSpec:
    name: str
    formula: Optional[str]
    coefficients: Optional[dict[str, float]]
    fn: Optional[Callable[..., np.ndarray]]
    noise_std: float


class PanelGenerator:
    """Builder for synthetic panel data with a known data-generating process.

    Parameters
    ----------
    n_entities:
        Number of cross-sectional units. Use ``n_entities=1`` for a pure
        time-series (no cross-sectional variation).
    n_periods:
        Number of time periods per entity.
    seed:
        Root seed. Same seed + same pinned spuriosity/numpy versions
        produces a byte-identical generated DataFrame. See
        docs/design_spec.md for the reproducibility contract.
    """

    def __init__(self, n_entities: int, n_periods: int, seed: int) -> None:
        if n_entities < 1:
            raise ValueError(f"n_entities must be >= 1, got {n_entities}")
        if n_periods < 1:
            raise ValueError(f"n_periods must be >= 1, got {n_periods}")

        self.n_entities = n_entities
        self.n_periods = n_periods
        self.seed = seed

        self._rng_manager = RNGManager(seed)
        self._variables: dict[str, _VariableSpec] = {}
        self._treatment: Optional[_TreatmentSpec] = None
        self._outcome: Optional[_OutcomeSpec] = None

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_variable(
        self,
        name: str,
        dist: str = "normal",
        **params,
    ) -> "PanelGenerator":
        """Declare a covariate drawn i.i.d. across entities and periods.

        Parameters
        ----------
        name:
            Column name. Must be a valid Python identifier and not already
            declared (as a variable, treatment, or outcome).
        dist:
            One of ``"normal"`` (params: ``mean``, ``std``) or ``"uniform"``
            (params: ``low``, ``high``).
        """
        self._check_name_available(name)
        if dist not in _SUPPORTED_DISTS:
            raise ValueError(f"Unsupported dist {dist!r}; supported: {_SUPPORTED_DISTS}")
        if dist == "normal":
            params.setdefault("mean", 0.0)
            params.setdefault("std", 1.0)
            if params["std"] < 0:
                raise ValueError(f"std must be >= 0, got {params['std']}")
        elif dist == "uniform":
            params.setdefault("low", 0.0)
            params.setdefault("high", 1.0)
            if params["low"] > params["high"]:
                raise ValueError(f"low ({params['low']}) must be <= high ({params['high']})")

        self._variables[name] = _VariableSpec(name=name, dist=dist, params=params)
        return self

    def add_treatment(
        self,
        name: str,
        assignment: str = "random",
        start_period: int = 0,
        propensity: float = 0.5,
    ) -> "PanelGenerator":
        """Declare a binary treatment indicator.

        Assignment is fixed per entity (not re-randomized across periods)
        and active from ``start_period`` onward for treated entities. This
        matches the standard panel/DiD treatment structure.

        Parameters
        ----------
        assignment:
            Currently only ``"random"`` is supported: each entity is
            independently treated with probability ``propensity``.
        start_period:
            First period (0-indexed) in which treatment takes effect for
            treated entities. Must be < n_periods.
        propensity:
            Probability an entity is treated, for ``assignment="random"``.
        """
        self._check_name_available(name)
        if assignment != "random":
            raise ValueError(f"Unsupported assignment {assignment!r}; only 'random' is supported in v1")
        if not (0 <= start_period < self.n_periods):
            raise ValueError(
                f"start_period must be in [0, n_periods)=[0, {self.n_periods}), got {start_period}"
            )
        if not (0.0 <= propensity <= 1.0):
            raise ValueError(f"propensity must be in [0, 1], got {propensity}")

        self._treatment = _TreatmentSpec(
            name=name, assignment="random", start_period=start_period, propensity=propensity
        )
        return self

    def set_outcome(
        self,
        formula: Optional[str] = None,
        coefficients: Optional[dict[str, float]] = None,
        fn: Optional[Callable[..., np.ndarray]] = None,
        name: str = "y",
        noise_std: float = 1.0,
    ) -> "PanelGenerator":
        """Define the outcome variable's data-generating process.

        Exactly one of ``formula`` or ``fn`` must be provided:

        - ``formula``: a patsy formula string for the right-hand side
          (e.g. ``"x1 + x2 + treat"``). Patsy builds the design matrix;
          ``coefficients`` supplies the true coefficient for each resulting
          column (including ``"Intercept"``, which patsy adds
          automatically). Columns not present in ``coefficients`` default
          to a true coefficient of 0.0.
        - ``fn``: a callable accepting the declared variable names (and the
          treatment name, if declared) as keyword arguments, returning the
          noiseless outcome as a length-N array. Use this for nonlinear or
          otherwise non-design-matrix DGPs.

        Gaussian noise with standard deviation ``noise_std`` is added on
        top of either path.
        """
        self._check_name_available(name)
        if (formula is None) == (fn is None):
            raise ValueError("Exactly one of `formula` or `fn` must be provided")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        if formula is not None and coefficients is None:
            raise ValueError("`coefficients` must be provided alongside `formula`")

        self._outcome = _OutcomeSpec(
            name=name, formula=formula, coefficients=coefficients, fn=fn, noise_std=noise_std
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_name_available(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"{name!r} is not a valid Python identifier")
        taken = set(self._variables) | {"entity_id", "period"}
        if self._treatment is not None:
            taken.add(self._treatment.name)
        if self._outcome is not None:
            taken.add(self._outcome.name)
        if name in taken:
            raise ValueError(f"Name {name!r} is already in use")

    def _draw_variable(self, spec: _VariableSpec, n: int) -> np.ndarray:
        gen = self._rng_manager.child("base_variables")
        if spec.dist == "normal":
            result: np.ndarray = gen.normal(loc=spec.params["mean"], scale=spec.params["std"], size=n)
            return result
        elif spec.dist == "uniform":
            result = gen.uniform(low=spec.params["low"], high=spec.params["high"], size=n)
            return result
        raise AssertionError(f"unreachable: unknown dist {spec.dist!r}")

    def _draw_treatment(self, n_rows: int, entity_ids: np.ndarray, periods: np.ndarray) -> np.ndarray:
        assert self._treatment is not None
        gen = self._rng_manager.child("treatment_assignment")
        entity_treated = gen.random(self.n_entities) < self._treatment.propensity
        row_entity_treated = entity_treated[entity_ids]
        active = periods >= self._treatment.start_period
        result: np.ndarray = (row_entity_treated & active).astype(int)
        return result

    def _compute_outcome_mean(self, data: pd.DataFrame) -> np.ndarray:
        assert self._outcome is not None
        if self._outcome.fn is not None:
            kwargs = {v: data[v].to_numpy() for v in self._variables}
            if self._treatment is not None:
                kwargs[self._treatment.name] = data[self._treatment.name].to_numpy()
            result = self._outcome.fn(**kwargs)
            return np.asarray(result, dtype=float)

        assert self._outcome.formula is not None
        assert self._outcome.coefficients is not None
        design = patsy.dmatrix(self._outcome.formula, data, return_type="dataframe")
        coefs = np.array(
            [self._outcome.coefficients.get(col, 0.0) for col in design.columns], dtype=float
        )
        outcome_mean: np.ndarray = design.to_numpy() @ coefs
        return outcome_mean

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> tuple[pd.DataFrame, GroundTruth]:
        """Generate the panel DataFrame and its GroundTruth.

        Requires at least `set_outcome(...)` to have been called; variables
        and treatment are optional (an outcome could, in principle, be a
        pure noise process via `fn=`).
        """
        if self._outcome is None:
            raise RuntimeError("set_outcome(...) must be called before generate()")

        n_rows = self.n_entities * self.n_periods
        entity_ids = np.repeat(np.arange(self.n_entities), self.n_periods)
        periods = np.tile(np.arange(self.n_periods), self.n_entities)

        data: dict[str, np.ndarray] = {"entity_id": entity_ids, "period": periods}

        for var_name, spec in self._variables.items():
            data[var_name] = self._draw_variable(spec, n_rows)

        df = pd.DataFrame(data)

        if self._treatment is not None:
            df[self._treatment.name] = self._draw_treatment(n_rows, entity_ids, periods)

        mean_outcome = self._compute_outcome_mean(df)
        noise_gen = self._rng_manager.child("outcome_noise")
        noise = noise_gen.normal(loc=0.0, scale=self._outcome.noise_std, size=n_rows)
        df[self._outcome.name] = mean_outcome + noise

        true_coefficients = dict(self._outcome.coefficients) if self._outcome.coefficients else {}
        treatment_effect_ate = (
            true_coefficients.get(self._treatment.name) if self._treatment is not None else None
        )

        truth = GroundTruth(
            true_coefficients=true_coefficients,
            treatment_effect_ate=treatment_effect_ate,
            spuriosity_version=_get_spuriosity_version(),
            numpy_version=np.__version__,
            seed=self.seed,
        )

        return df, truth


def _get_spuriosity_version() -> str:
    from spuriosity import __version__

    return __version__
