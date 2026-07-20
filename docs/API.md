# spuriosity API reference

**This document is verified, not aspirational.** Every import, every
signature, and every example in this file is exercised by
[`scripts/verify_api_doc.py`](../scripts/verify_api_doc.py), which is run
before any change to this document is committed. If something here looks
wrong, the verification script — and the actual installed package — are
the source of truth, not this prose.

```
Verified against: spuriosity 0.1.0
Last verification run: 14/14 checks passed
```

To re-verify yourself:

```bash
pip install -e ".[sklearn,viz,linearmodels,doubleml,xgboost,econml]"
python scripts/verify_api_doc.py
```

---

## Installation

```bash
# Core only (patsy, statsmodels, numpy, pandas, scipy)
pip install -e .

# With everything used across the reference toolbox and viz
pip install -e ".[sklearn,viz,linearmodels,doubleml,xgboost,econml]"
```

From GitHub (e.g. in a Colab cell — use `%pip`, not `!pip`, so the install
targets the running kernel's own environment):

```python
%pip install -q "spuriosity[sklearn,viz,linearmodels,doubleml,xgboost,econml] @ git+https://github.com/Nityahapani/spuriosity.git"
```

## Optional dependency groups

| Extra | Package | Required by |
|---|---|---|
| `sklearn` | `scikit-learn` | `reference.sklearn_lr_fit`*, `reference.psm_fit` |
| `viz` | `matplotlib` | `plot_recovery_report` |
| `linearmodels` | `linearmodels` | `reference.iv2sls_fit`, `reference.panel_fe_fit`, `reference.panel_re_fit` |
| `doubleml` | `doubleml` | `reference.doubleml_fit` |
| `xgboost` | `xgboost` | `reference.xgboost_fit` |
| `econml` | `econml` (pulls in `numba`, `lightgbm`, `shap`) | `reference.causal_forest_fit` |

\* `sklearn_lr_fit` imports `sklearn` directly without a guarded
`ImportError` (unlike every other optional-dependency function below) —
a known minor inconsistency; if `scikit-learn` isn't installed, calling
it raises a raw `ModuleNotFoundError` rather than a `spuriosity`-authored
message pointing at `pip install spuriosity[sklearn]`.

Every other optional-dependency function above raises a clear
`ImportError` naming the missing package and the exact `pip install`
command, if called without the dependency installed. `import spuriosity`
itself never requires any optional dependency — only calling a specific
function that needs one does.

**First-call cost for `causal_forest_fit`**: `econml` depends on `numba`,
which performs a one-time LLVM/JIT compilation cache build on first
import in a given environment (observed ~15-20s cold, ~2s warm). This is
a one-time environment cost, not a per-call cost.

---

## Top-level exports (`import spuriosity`)

Every name below is importable as `from spuriosity import <name>`, and is
listed in `spuriosity.__all__` (verified — the two are checked for exact
equality, not just "documented names exist").

### Generation

| Name | Kind | Purpose |
|---|---|---|
| `PanelGenerator` | class | The builder — construct a panel DGP, then `.generate()` |
| `GroundTruth` | class (frozen dataclass) | Returned alongside every generated DataFrame |
| `HTE` | class | Heterogeneous treatment effect spec (used internally by `add_hte`) |
| `Pathology` | abstract class | Base class all pathologies subclass |
| `StructuralBreak` | class | Regime change pathology |
| `Confounder` | class | Omitted-variable-bias pathology |
| `SelectionBias` | class | Non-random sample selection pathology |
| `Heteroskedasticity` | class | Non-constant error variance pathology |
| `Multicollinearity` | class | Correlated-regressor pathology |
| `MeasurementError` | class | Classical errors-in-variables pathology |
| `Endogeneity` | class | Regressor-correlated-with-error pathology, with instrument |
| `UnitRoot` | class | Random-walk (nonstationarity) pathology |

### Ground truth sub-records

| Name | Kind |
|---|---|
| `BreakInfo` | frozen dataclass |
| `SelectionInfo` | frozen dataclass |
| `HeteroskedasticityInfo` | frozen dataclass |
| `MulticollinearityInfo` | frozen dataclass |
| `MeasurementErrorInfo` | frozen dataclass |
| `EndogeneityInfo` | frozen dataclass |
| `UnitRootInfo` | frozen dataclass |

### Evaluation

| Name | Kind | Purpose |
|---|---|---|
| `reference` | module | Batteries-included fit/predict functions (see below) |
| `StressTest` | class | Evaluate one model against a `GroundTruth` |
| `StressTestReport` | class | Result of `StressTest.evaluate()` |
| `compare_models` | function | Evaluate several models against the same `GroundTruth` |
| `ComparisonReport` | class | Result of `compare_models()` |
| `MetricContext` | class (frozen dataclass) | Passed to every registered metric function |
| `MetricRegistry` | class | Pluggable metric registration |
| `default_registry` | `MetricRegistry` instance | The four built-in metrics |
| `plot_recovery_report` | function | Visualize a `StressTestReport` or `ComparisonReport` |
| `synthetic_control_fit` | function | Synthetic control method + placebo-in-space inference |
| `SyntheticControlResult` | class | Result of `synthetic_control_fit()` |

---

## `PanelGenerator`

```python
PanelGenerator(n_entities: int, n_periods: int, seed: int)
```

`n_entities=1` gives a pure time series (no cross-sectional variation).

### Builder methods (all return `self`, chainable)

```python
add_variable(name: str, dist: str = "normal", **params) -> PanelGenerator
```
`dist="normal"` takes `mean`, `std`. `dist="uniform"` takes `low`, `high`.

```python
add_treatment(
    name: str,
    assignment: str = "random",       # "random" | "propensity"
    start_period: int = 0,
    propensity: float = 0.5,          # used by assignment="random"
    propensity_formula: str | None = None,  # required by assignment="propensity"
) -> PanelGenerator
```
Treatment is fixed per entity across periods, active from `start_period`
onward. `assignment="propensity"` makes assignment covariate-dependent:
probability is `sigmoid(propensity_formula)`, evaluated once per entity
at period 0 using that entity's covariate values (`propensity_formula`
may reference any `add_variable`-declared name, evaluated via
`pandas.eval` — see the security note under `add_selection_bias` below).

```python
set_outcome(
    formula: str | None = None,
    coefficients: dict[str, float] | None = None,  # required with formula=
    fn: Callable | None = None,       # exactly one of formula/fn required
    name: str = "y",
    noise_std: float = 1.0,
) -> PanelGenerator
```
`formula` is a patsy right-hand-side expression (e.g. `"x1 + x2 + treat"`);
patsy builds the design matrix (including the automatic `"Intercept"`
column), and `coefficients` supplies the *true* coefficient for each
resulting column — patsy formulas describe structure, not magnitudes.
`fn` is a callable taking declared variable/treatment names as keyword
arguments, for DGPs too nonlinear for a linear design matrix.

```python
add_structural_break(
    period: int, target: str,
    kind: str = "mean_shift",   # "mean_shift" | "variance_shift" | "coefficient_shift"
    magnitude: float = 0.0,
    coefficient_target: str | None = None,  # required for kind="coefficient_shift"
) -> PanelGenerator

add_confounder(feature: str, outcome: str, strength: float, observed: bool = False) -> PanelGenerator
add_selection_bias(rule: str, drop_prob: float) -> PanelGenerator
add_heteroskedasticity(feature: str, formula: str) -> PanelGenerator
add_multicollinearity(feature: str, correlated_with: str, correlation: float) -> PanelGenerator
add_measurement_error(feature: str, noise_std: float) -> PanelGenerator
add_endogeneity(
    feature: str, instrument: str,
    instrument_strength: float, endogeneity_strength: float,
    first_stage_noise_std: float = 0.5,
) -> PanelGenerator
add_unit_root(feature: str, drift: float = 0.0) -> PanelGenerator
add_hte(treatment: str, modifier: str | list[str], formula: str) -> PanelGenerator
```

**Naming convention split** — worth knowing before combining pathologies:
- **Modifies an existing column** (must already be declared via
  `add_variable`): `add_confounder`, `add_heteroskedasticity`,
  `add_measurement_error`.
- **Creates a new column** (name must NOT already be declared):
  `add_multicollinearity` (its `feature`), `add_endogeneity` (its
  `instrument`).
- `add_selection_bias`'s `rule` and `add_hte`'s `formula` (and
  `propensity_formula` above) are evaluated via a constrained
  `pandas.eval` (explicit `local_dict`, `global_dict={}`, `engine="python"`)
  — arbitrary Python function calls (e.g. `__import__(...)`) are rejected
  by this grammar, not merely discouraged. See `CONTRIBUTING.md` for the
  full security stance.

```python
validate_combo() -> list[str]
```
Checks added pathologies for likely conflicts. **Prints warnings and
returns them as a list — never raises.** Permissive by design.

```python
generate() -> tuple[pandas.DataFrame, GroundTruth]
```
Requires `set_outcome()` to have been called first.

### Verified example

```python
from spuriosity import PanelGenerator

gen = PanelGenerator(n_entities=5000, n_periods=10, seed=42)
gen.add_variable("x1", dist="normal", mean=0, std=1)
gen.add_treatment("treat", assignment="random", start_period=5, propensity=0.5)
gen.set_outcome(formula="x1 + treat", coefficients={"x1": 2.0, "treat": 3.0}, noise_std=0.5)
gen.add_confounder(feature="x1", outcome="y", strength=0.3, observed=True)
df, truth = gen.generate()
```

### A gotcha worth knowing: DiD-style scenarios

If you want treatment **group membership** fixed from period 0 but the
**effect** to only begin at a later period (the standard DiD setup), use
`add_treatment(start_period=0, ...)` for group membership and a separate
`add_structural_break(kind="coefficient_shift", ...)` for when the effect
kicks in. Using `add_treatment`'s own `start_period` to mean "when the
effect begins" makes the treatment column identical to
`treatment_group * post` already — which makes it **perfectly collinear**
with `reference.did_fit`'s own `treatment * post` interaction term
(observed directly during notebook testing: an OLS condition number of
~10¹⁴ and a nonsensical ~10¹² coefficient). See
[`examples/README.md`](../examples/README.md) for the worked version of
this.

---

## `GroundTruth`

Frozen dataclass, returned by `PanelGenerator.generate()`. **14 fields**
(verified exhaustively against the installed dataclass):

```python
true_coefficients: dict[str, float]
break_points: list[BreakInfo]
confounding_strength: dict[str, float]
true_cate: Callable[[float], float] | None       # or Callable[..., float] for multi-dim HTE
selection_mechanism: SelectionInfo | None
heteroskedasticity: list[HeteroskedasticityInfo]
multicollinearity: list[MulticollinearityInfo]
measurement_error: list[MeasurementErrorInfo]
endogeneity: list[EndogeneityInfo]
unit_root: list[UnitRootInfo]
treatment_effect_ate: float | None
spuriosity_version: str
numpy_version: str
seed: int
```

Methods: `.to_dict()`, `.to_json(indent=2)`, `.__repr__()` (compact,
notebook-friendly summary).

**Multi-dimensional HTE**: if `add_hte(modifier=["x1","x2"])` was used
(2+ modifiers), `true_cate` must be called with **keyword arguments**
matching each modifier's name (`truth.true_cate(x1=1.0, x2=2.0)`), not
positionally — a positional call raises `TypeError`. Single-modifier HTE
keeps the original positional signature (`truth.true_cate(1.0)`)
unchanged.

---

## `spuriosity.reference`

Every fit function returns a `FitResult`:

```python
@dataclass
class FitResult:
    coefficients: dict[str, float]   # matches GroundTruth naming where applicable
    raw_model: Any                   # the underlying fitted object
    ate_estimate: float | None
    ate_std_error: float | None
    extra: dict[str, Any]            # fit-function-specific metadata
```

### The 14 fit/predict pairs (+ 3 standalone helpers) — 24 symbols total

| Fit function | Predict function | Notes |
|---|---|---|
| `ols_fit(data, formula)` | `ols_predict(fit, data)` | statsmodels OLS. No optional dep. |
| `sklearn_lr_fit(data, features, target)` | `sklearn_lr_predict(fit, data)` | sklearn LinearRegression |
| `logit_fit(data, formula)` | `logit_predict(fit, data)` | Returns probabilities, not labels |
| `did_fit(data, outcome, treatment, period, post_period)` | `did_predict(fit, data)` | `.ate_estimate` is the `treatment:post` coefficient |
| `panel_fe_fit(data, outcome, features, entity_col="entity_id", period_col="period")` | `panel_fe_predict(fit, data)` | No intercept; **cannot estimate a time-invariant regressor** (fully absorbed — raises `AbsorbingEffectError`) |
| `panel_re_fit(data, outcome, features, entity_col="entity_id", period_col="period")` | `panel_re_predict(fit, data)` | Intercept under linearmodels' `"const"` key, not patsy's `"Intercept"` |
| `iv2sls_fit(data, outcome, endogenous, instruments, exogenous=None)` | `iv2sls_predict(fit, data)` | 2SLS via linearmodels |
| `doubleml_fit(data, outcome, treatment, covariates, n_folds=5)` | `doubleml_predict(fit, data)` | `.ate_estimate`; predict broadcasts the ATE to every row |
| `psm_fit(data, outcome, treatment, covariates)` | `psm_predict(fit, data)` | Nearest-neighbor propensity matching; `.extra["common_support_fraction"]` |
| `xgboost_fit(data, outcome, features, n_estimators=100, max_depth=4, **xgb_kwargs)` | `xgboost_predict(fit, data)` | `.coefficients` deliberately empty; see `.extra["feature_importances"]` |
| `causal_forest_fit(data, outcome, treatment, covariates, n_estimators=200, discrete_treatment=True, random_state=None)` | `causal_forest_predict(fit, data)` | **`causal_forest_predict` returns per-row CATE, not an outcome prediction** — the one function that breaks the usual predict convention. `n_estimators` must be divisible by 4. |

Standalone helpers (not fit/predict pairs):

```python
first_stage_f_stat(data, endogenous: str, instruments: list[str]) -> float
hausman_test(fe_result: FitResult, re_result: FitResult) -> dict[str, float]  # {"chi2", "dof", "p_value"}
```

### Verified example

```python
from spuriosity import PanelGenerator, reference

gen = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
gen.add_variable("x1")
gen.add_treatment("treat", propensity=0.5)
gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 2.0}, noise_std=0.5)
df, truth = gen.generate()

fit = reference.ols_fit(df, formula="y ~ x1 + treat")
preds = reference.ols_predict(fit, df)
```

---

## `StressTest` / `compare_models`

```python
StressTest(truth: GroundTruth, metric_registry: MetricRegistry | None = None)

StressTest.evaluate(
    fit_fn: Callable, predict_fn: Callable, data: pd.DataFrame,
    fit_kwargs: dict | None = None, model_name: str = "model",
    period_col: str = "period",
) -> StressTestReport
```

`metric_registry` defaults to `spuriosity.default_registry` (four
built-ins: `coef_rmse`, `confounding_bias`, `break_detection_lag`,
`cate_rmse` — the last is a defined slot, not yet implemented, and never
appears in results).

```python
compare_models(
    data: pd.DataFrame, truth: GroundTruth,
    models: dict[str, tuple[Callable, Callable]],
    weights: dict[str, float] | None = None,
    fit_kwargs_per_model: dict[str, dict] | None = None,
    metric_registry: MetricRegistry | None = None,
) -> ComparisonReport

ComparisonReport.ranked_table(by: str = "default_composite") -> pd.DataFrame
ComparisonReport.summary() -> None
```

`ranked_table` tracks excluded models (those missing the requested
metric) via `.attrs["excluded_models"]` — never silently scores them as
0.

### Custom metrics

```python
from spuriosity import PanelGenerator, StressTest, MetricRegistry, default_registry
from spuriosity.metrics import MetricContext

gen = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
gen.add_variable("x1")
gen.set_outcome(formula="x1", coefficients={"x1": 2.0})
df, truth = gen.generate()

def my_metric(ctx: MetricContext) -> float | dict[str, float] | None:
    return 1.0  # return None if not applicable to this fit

registry = default_registry.copy()
registry.register("my_metric", my_metric)
test = StressTest(truth, metric_registry=registry)
```

`MetricContext` fields: `truth`, `fitted_coefficients`, `fit_result`,
`data`, `fit_fn`, `fit_kwargs`, `period_col`.

### Verified example

```python
from spuriosity import PanelGenerator, StressTest, compare_models, reference

gen = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
gen.add_variable("x1")
gen.set_outcome(formula="x1", coefficients={"x1": 2.0})
df, truth = gen.generate()

results = compare_models(
    data=df, truth=truth,
    models={"OLS": (reference.ols_fit, reference.ols_predict)},
    fit_kwargs_per_model={"OLS": {"formula": "y ~ x1"}},
)
results.summary()
```

---

## `plot_recovery_report`

```python
plot_recovery_report(
    report: StressTestReport | ComparisonReport,
    cate_range: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> matplotlib.figure.Figure
```

Requires `pip install spuriosity[viz]` (lazy-imported; `import
spuriosity` itself never requires matplotlib). `cate_range` only supports
**single-dimension** `true_cate` — passing it for a multi-dimensional HTE
report raises a clear `ValueError` rather than a confusing internal
`TypeError`.

---

## `synthetic_control_fit`

```python
synthetic_control_fit(
    data: pd.DataFrame, outcome: str, entity_col: str, period_col: str,
    treated_unit: object, treatment_period: object,
    donor_units: list | None = None,
    run_placebo_inference: bool = True,
) -> SyntheticControlResult
```

Own module: `from spuriosity.synthetic_control import synthetic_control_fit`
(also re-exported at the top level as `spuriosity.synthetic_control_fit`).

`SyntheticControlResult` fields: `treated_unit`, `weights` (dict, donor →
weight, nonnegative and summing to 1), `periods`, `treatment_period`,
`treated_outcome`, `synthetic_outcome`, `effect_by_period`,
`average_effect`, `pre_period_fit_rmse`, `placebo_effects` (dict or
`None`), `placebo_p_value` (float or `None`). Method: `.summary()`.

**Placebo-in-space caveat**: because each placebo unit's own donor pool
includes the *real* treated unit, a donor with substantial weight on the
treated unit can itself produce a large apparent "placebo effect" —
occasionally exceeding the real effect in magnitude, especially with a
small donor pool. The resulting p-value is not guaranteed to hit its
theoretical floor (`1/(k+1)` for `k` donors) for every effect size; it
reliably approaches that floor with a larger, more diverse pool. This is
a documented property of the method in the literature, not a bug — see
`docs/design_spec.md` for the full account of how this was found during
testing.

### Verified example

```python
from spuriosity import synthetic_control_fit
import pandas as pd
import numpy as np

# Build a small panel: entity 0 is treated from period 10 onward.
rng = np.random.default_rng(1)
common_factor = rng.normal(size=20).cumsum() * 0.3
rows = []
for entity in range(10):
    loading = rng.uniform(0.5, 1.5)
    for period in range(20):
        rows.append({"entity_id": entity, "period": period, "y": common_factor[period] * loading})
df = pd.DataFrame(rows)

result = synthetic_control_fit(
    df, outcome="y", entity_col="entity_id", period_col="period",
    treated_unit=0, treatment_period=10,
)
result.summary()
```

---

## Verification

Everything above is checked by
[`scripts/verify_api_doc.py`](../scripts/verify_api_doc.py):

```bash
python scripts/verify_api_doc.py
```

The script checks, against the actually-installed package:
- `spuriosity.__all__` matches the documented symbol list exactly (no
  missing, no extra)
- Every symbol is importable both via `spuriosity.X` and
  `from spuriosity import X`
- `spuriosity.reference` exposes exactly the 24 documented symbols
- `GroundTruth`'s dataclass fields match the documented 14 fields exactly
- Every `PanelGenerator` builder method runs in a full chained example
- Every one of the 14 reference fits runs against real generated data
- `hausman_test`, `first_stage_f_stat`, `StressTest`, `compare_models`,
  custom `MetricRegistry`, `plot_recovery_report`, and
  `synthetic_control_fit` each run end-to-end

If this document and the script ever disagree, trust the script's
output — it reflects what the installed code actually does, not what
someone (including Claude) remembered or assumed while writing prose.
