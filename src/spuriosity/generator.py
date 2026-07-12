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

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import patsy

from spuriosity._rng import RNGManager
from spuriosity.ground_truth import GroundTruth
from spuriosity.hte import HTE
from spuriosity.pathologies import Confounder, SelectionBias, StructuralBreak, validate_combo as _validate_combo

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
        self._pathologies: list = []
        self._hte: Optional[HTE] = None

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

    def add_hte(self, treatment: str, modifier: str, formula: str) -> "PanelGenerator":
        """Make `treatment`'s effect on the outcome vary with `modifier`,
        according to `formula` (a pandas.eval-evaluated expression in terms
        of `modifier`, e.g. ``"3 + 1.5*x1"``).

        Requires a treatment declared via `add_treatment` and an outcome
        specified via `set_outcome(formula=...)` (HTE is not supported with
        an `fn=`-specified outcome in v1, since it needs to isolate and
        replace the treatment's linear term in the design matrix).

        See `spuriosity.hte.HTE` for the exact mechanism: this *replaces*
        the treatment's contribution to the outcome entirely, ignoring any
        coefficient given for it in `set_outcome(coefficients=...)`.
        """
        self._hte = HTE(treatment=treatment, modifier=modifier, formula=formula)
        return self

    def add_structural_break(
        self,
        period: int,
        target: str,
        kind: str = "mean_shift",
        magnitude: float = 0.0,
        coefficient_target: Optional[str] = None,
    ) -> "PanelGenerator":
        """Inject a structural break (regime change) at `period`.

        See `spuriosity.pathologies.StructuralBreak` for the semantics of
        each `kind` ("mean_shift", "variance_shift", "coefficient_shift").
        `target` should normally be the outcome's name; a warning-worthy
        mismatch is not currently checked, since `target` is only used for
        ground-truth bookkeeping.
        """
        if not (0 <= period < self.n_periods):
            raise ValueError(f"period must be in [0, n_periods)=[0, {self.n_periods}), got {period}")
        break_pathology = StructuralBreak(
            period=period,
            target=target,
            kind=kind,  # type: ignore[arg-type]
            magnitude=magnitude,
            coefficient_target=coefficient_target,
        )
        self._pathologies.append(break_pathology)
        return self

    def add_confounder(
        self,
        feature: str,
        outcome: str,
        strength: float,
        observed: bool = False,
    ) -> "PanelGenerator":
        """Inject a latent confounder `U` that causally affects both
        `feature` and `outcome`, inducing omitted-variable bias in a naive
        regression that omits `U`.

        See `spuriosity.pathologies.Confounder` for the exact mechanism and
        the closed-form predicted bias formula.

        `feature` must already be declared via `add_variable` or
        `add_treatment` -- the confounder modifies an existing column, it
        does not create one. This is checked at `generate()` time (once all
        builder calls are known), not here, so `add_confounder` may be
        called before or after the corresponding `add_variable` call.

        If `observed=True`, `U` is exposed as a visible column named
        `f"_confounder_{feature}"` in the generated DataFrame.
        """
        confounder = Confounder(feature=feature, outcome=outcome, strength=strength, observed=observed)
        self._pathologies.append(confounder)
        return self

    def add_selection_bias(self, rule: str, drop_prob: float) -> "PanelGenerator":
        """Apply non-random sample selection: rows matching `rule` are
        dropped with probability `drop_prob`.

        See `spuriosity.pathologies.SelectionBias` for the rule evaluation
        mechanism (a constrained `pandas.eval`, see CONTRIBUTING.md) and
        the security stance around it. `rule` may reference any column
        present in the generated DataFrame, including the outcome, so
        outcome-dependent selection (survivorship bias) can be modeled.
        """
        self._pathologies.append(SelectionBias(rule=rule, drop_prob=drop_prob))
        return self

    def validate_combo(self) -> list[str]:
        """Check currently-added pathologies for likely conflicts.

        Prints any warnings found and also returns them as a list. Never
        raises — v1 policy is permissive by design (see docs/design_spec.md).
        """
        warnings_found = _validate_combo(self._pathologies)
        for w in warnings_found:
            print(f"[spuriosity warning] {w}")
        return warnings_found

    def __repr__(self) -> str:
        """Compact, debugging-friendly summary of the builder state.

        Not a round-trip repr (re-evaluating it will not rebuild the
        generator -- this is a snapshot, not a recipe). For a serializable
        record of the DGP, use the `GroundTruth` returned by `generate()`.
        """
        parts: list[str] = [
            f"PanelGenerator(n_entities={self.n_entities}, n_periods={self.n_periods}, seed={self.seed})",
        ]

        # Variables
        if self._variables:
            var_summary = ", ".join(
                f"{name}({spec.dist})" for name, spec in self._variables.items()
            )
            parts.append(f"  variables: {var_summary}")
        else:
            parts.append("  variables: <none>")

        # Treatment
        if self._treatment is not None:
            t = self._treatment
            parts.append(
                f"  treatment: {t.name}(assignment={t.assignment}, "
                f"start_period={t.start_period}, propensity={t.propensity})"
            )
        else:
            parts.append("  treatment: <none>")

        # Outcome
        if self._outcome is not None:
            o = self._outcome
            kind = "fn" if o.fn is not None else f"formula={o.formula!r}"
            coefs = (
                "{" + ", ".join(f"{k!r}: {v}" for k, v in (o.coefficients or {}).items()) + "}"
                if o.coefficients
                else "<none>"
            )
            parts.append(f"  outcome: name={o.name!r}, {kind}, coefficients={coefs}, noise_std={o.noise_std}")
        else:
            parts.append("  outcome: <unset -- call set_outcome(...) before generate()>")

        # Pathologies (compact by type)
        if self._pathologies:
            by_type: dict[str, int] = {}
            for p in self._pathologies:
                by_type[type(p).__name__] = by_type.get(type(p).__name__, 0) + 1
            path_summary = ", ".join(f"{k}×{v}" for k, v in sorted(by_type.items()))
            parts.append(f"  pathologies ({len(self._pathologies)}): {path_summary}")
        else:
            parts.append("  pathologies: <none>")

        # HTE
        if self._hte is not None:
            h = self._hte
            parts.append(
                f"  hte: treatment={h.treatment!r}, modifier={h.modifier!r}, formula={h.formula!r}"
            )
        else:
            parts.append("  hte: <none>")

        return "\n".join(parts)

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

    def _compute_outcome_mean(self, data: pd.DataFrame) -> tuple[np.ndarray, Optional[pd.DataFrame]]:
        """Returns (mean_outcome, design_matrix_or_None). design_matrix is
        only produced for the formula path (needed by coefficient_shift
        structural breaks); it is None for the fn= path."""
        assert self._outcome is not None
        if self._outcome.fn is not None:
            kwargs = {v: data[v].to_numpy() for v in self._variables}
            if self._treatment is not None:
                kwargs[self._treatment.name] = data[self._treatment.name].to_numpy()
            result = self._outcome.fn(**kwargs)
            return np.asarray(result, dtype=float), None

        assert self._outcome.formula is not None
        assert self._outcome.coefficients is not None
        design = patsy.dmatrix(self._outcome.formula, data, return_type="dataframe")
        coefs = np.array(
            [self._outcome.coefficients.get(col, 0.0) for col in design.columns], dtype=float
        )
        outcome_mean: np.ndarray = design.to_numpy() @ coefs
        return outcome_mean, design

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

        confounders = [p for p in self._pathologies if isinstance(p, Confounder)]
        confounder_outcome_contribution = np.zeros(n_rows, dtype=float)
        for conf in confounders:
            if conf.feature not in df.columns:
                raise ValueError(
                    f"Confounder targets feature {conf.feature!r}, which is not a declared "
                    f"variable or treatment on this PanelGenerator."
                )
            if self._treatment is not None and conf.feature == self._treatment.name:
                warnings.warn(
                    f"Confounder targets the treatment column {conf.feature!r}. This will turn "
                    "it into a continuous variable (no longer strictly 0/1), which may silently "
                    "break estimators that assume a binary treatment. See "
                    "spuriosity.pathologies.Confounder docstring for details.",
                    stacklevel=2,
                )
            gen = self._rng_manager.child(f"confounder:{conf.feature}")
            new_feature, contribution, u = conf.draw_and_apply(
                df[conf.feature].to_numpy(), confounder_outcome_contribution, gen
            )
            df[conf.feature] = new_feature
            confounder_outcome_contribution = contribution
            if conf.observed:
                df[f"_confounder_{conf.feature}"] = u

        mean_outcome, design = self._compute_outcome_mean(df)
        mean_outcome = mean_outcome + confounder_outcome_contribution

        true_cate_fn = None
        if self._hte is not None:
            if self._outcome.fn is not None:
                raise ValueError(
                    "add_hte() requires an outcome specified via set_outcome(formula=...); "
                    "it is not supported with an fn=-specified outcome."
                )
            if self._treatment is None or self._hte.treatment != self._treatment.name:
                raise ValueError(
                    f"add_hte() targets treatment {self._hte.treatment!r}, but no treatment "
                    f"with that name was declared via add_treatment()."
                )
            if self._hte.modifier not in df.columns:
                raise ValueError(
                    f"add_hte() modifier {self._hte.modifier!r} is not a declared variable, "
                    f"treatment, or reserved column on this PanelGenerator."
                )
            assert design is not None
            if self._hte.treatment not in design.columns:
                raise ValueError(
                    f"Treatment {self._hte.treatment!r} does not appear as a term in the outcome "
                    f"formula {self._outcome.formula!r}; add_hte() needs the treatment to be an "
                    "explicit additive term (e.g. include it in set_outcome(formula=...))."
                )

            fixed_treatment_coef = (self._outcome.coefficients or {}).get(self._hte.treatment, 0.0)
            if fixed_treatment_coef != 0.0:
                warnings.warn(
                    f"add_hte() replaces the effect of treatment {self._hte.treatment!r} entirely; "
                    f"the fixed coefficient {fixed_treatment_coef} supplied in "
                    "set_outcome(coefficients=...) for this treatment is being ignored.",
                    stacklevel=2,
                )

            treatment_col = design[self._hte.treatment].to_numpy()
            # Remove the (possibly zero) fixed linear contribution patsy/coefficients
            # would otherwise have added for the treatment term.
            mean_outcome = mean_outcome - fixed_treatment_coef * treatment_col

            per_row_effect = self._hte.evaluate_on_column(df[self._hte.modifier].to_numpy())
            mean_outcome = mean_outcome + per_row_effect * treatment_col
            true_cate_fn = self._hte.cate_fn()

        structural_breaks = [p for p in self._pathologies if isinstance(p, StructuralBreak)]
        for brk in structural_breaks:
            mean_outcome = brk.apply_to_mean(mean_outcome, periods, design, self._outcome.coefficients)

        noise_std_per_row = np.full(n_rows, self._outcome.noise_std, dtype=float)
        for brk in structural_breaks:
            if brk.kind == "variance_shift":
                shifted = brk.apply_to_noise_std(self._outcome.noise_std, periods)
                # Compose multiplicatively if multiple variance shifts overlap.
                factor = np.divide(
                    shifted, self._outcome.noise_std, out=np.ones_like(shifted), where=self._outcome.noise_std != 0
                )
                noise_std_per_row = noise_std_per_row * factor

        noise_gen = self._rng_manager.child("outcome_noise")
        noise = noise_gen.normal(loc=0.0, scale=1.0, size=n_rows) * noise_std_per_row
        df[self._outcome.name] = mean_outcome + noise

        selection_biases = [p for p in self._pathologies if isinstance(p, SelectionBias)]
        for i, sel in enumerate(selection_biases):
            sel_gen = self._rng_manager.child(f"selection_bias:{i}:{sel.rule}")
            drop_mask = sel.compute_mask_to_drop(df, sel_gen)
            df = df[~drop_mask].reset_index(drop=True)

        if len(selection_biases) > 1:
            warnings.warn(
                f"{len(selection_biases)} selection_bias pathologies were added; all are applied "
                "sequentially to the generated data, but only the first is recorded in "
                "GroundTruth.selection_mechanism (v1 supports a single selection mechanism in "
                "ground truth bookkeeping).",
                stacklevel=2,
            )

        true_coefficients = dict(self._outcome.coefficients) if self._outcome.coefficients else {}
        treatment_effect_ate: Optional[float]
        if self._hte is not None:
            treatment_effect_ate = float(np.mean(per_row_effect))
        else:
            treatment_effect_ate = (
                true_coefficients.get(self._treatment.name) if self._treatment is not None else None
            )

        break_points = []
        for brk in structural_breaks:
            break_points.extend(brk.ground_truth_contribution()["break_points"])

        confounding_strength = {}
        for conf in confounders:
            confounding_strength.update(conf.ground_truth_contribution()["confounding_strength"])

        selection_mechanism = None
        if selection_biases:
            selection_mechanism = selection_biases[0].ground_truth_contribution()["selection_mechanism"]

        truth = GroundTruth(
            true_coefficients=true_coefficients,
            break_points=break_points,
            confounding_strength=confounding_strength,
            true_cate=true_cate_fn,
            selection_mechanism=selection_mechanism,
            treatment_effect_ate=treatment_effect_ate,
            spuriosity_version=_get_spuriosity_version(),
            numpy_version=np.__version__,
            seed=self.seed,
        )

        return df, truth


def _get_spuriosity_version() -> str:
    from spuriosity import __version__

    return __version__
