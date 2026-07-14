# spuriosity — v1 Design Spec

## Concept

`spuriosity` generates synthetic panel/time-series data with a **known data-generating
process (DGP)**, then lets the user deliberately inject named econometric pathologies
into it. Every generated dataset ships with the ground truth, so any model or pipeline
can be run against it and scored on exactly how much a pathology fooled it.

This differs from realism-oriented synthetic data tools (SDV, Gretel, Synthea), which
optimize for matching a real dataset's distribution for privacy-safe sharing or
prototyping. `spuriosity` optimizes for the opposite: known, controllable failure modes,
for robustness testing and causal-inference benchmarking.

## Scope (v1)

- **Structure**: Panel data only. `n_entities=1` covers the time-series case — one
  structure serves both, no separate time-series API.
- **Pathologies (v1)**: Structural break, confounding, selection bias.
  - Unit root deferred to v1.1 (already well-served by statsmodels).
  - Multi-dimensional HTE modifiers deferred to v1.1 (v1 ships single-dimension only).

## Object model

- `PanelGenerator` — builder. Holds entities × periods, variables, outcome formula,
  treatment, HTE spec, and a list of stacked pathologies.
- `Pathology` (base class) → `StructuralBreak`, `Confounder`, `SelectionBias`.
  Each pathology knows how to (a) modify the DGP and (b) report the ground truth
  it changed.
- `GroundTruth` — frozen dataclass returned alongside the generated data.
- `StressTest` — evaluates a single fitted model/estimator against ground truth.
- `compare_models()` — runs multiple models against the same DGP, produces a
  ranked benchmark report.
- `plot_recovery_report()` — first-class visualization for both single-model and
  multi-model reports.
- `spuriosity.reference` — batteries-included fit functions (OLS, sklearn LR, DiD,
  DoubleML, logit) so new users get a working stress test in ~5 lines.

## Key decisions

| Area | Decision |
|---|---|
| Structure | Panel only (`n_entities=1` = time-series case) |
| v1 pathologies | Structural break, confounding, selection bias |
| Composability | Permissive. `validate_combo()` warns on likely-conflicting combos, never raises. |
| Output format | pandas only (no polars in v1) |
| GroundTruth | Frozen dataclass, not Pydantic. `.to_dict()` / `.to_json()`. |
| Formula language | [patsy](https://patsy.readthedocs.io/) strings for the 80% case; Python callables for the 20% complex/nonlinear case. |
| Selection rule | Arbitrary boolean via `pandas.eval`-style strings, with explicit `local_dict`/`global_dict` (no arbitrary global namespace access). Security stance documented in `CONTRIBUTING.md`. |
| HTE | Single-dimension modifier only in v1. `GroundTruth.true_cate` is a clean 1D callable. |
| StressTest | Function-based (`fit_fn`, `predict_fn`) for flexibility, with `spuriosity.reference` providing common fits out of the box. |
| `compare_models` ranking | Default composite score = weighted sum of relevant component metrics (weights default to 1.0, fully user-overridable). Component metrics only computed if the relevant pathology/HTE is present. Individual metrics are always exposed, never hidden inside the composite. `ranked_table(by=...)` supports ranking by the composite or any single metric. |
| Reproducibility | `seed` + pinned `spuriosity` + `numpy` versions → byte-identical DataFrame. **No cross-version guarantee.** One global `numpy.random.default_rng(seed)`, with sub-streams spawned per pathology (not shared sequential draws) to avoid order-dependence. Versions and seed stored in `GroundTruth`. |
| License | MIT |
| Python floor | 3.10+ |

## `GroundTruth` shape (v1)

```python
@dataclass(frozen=True)
class GroundTruth:
    true_coefficients: dict[str, float]
    break_points: list[BreakInfo]
    confounding_strength: dict[str, float]
    true_cate: Callable[[float], float] | None
    selection_mechanism: SelectionInfo | None
    treatment_effect_ate: float | None
    spuriosity_version: str
    numpy_version: str
    seed: int

    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
```

## Example API (target v1 surface)

```python
from spuriosity import PanelGenerator, StressTest, compare_models, plot_recovery_report
from spuriosity import reference

# 1. Base DGP
gen = PanelGenerator(n_entities=500, n_periods=40, seed=42)
gen.add_variable("x1", dist="normal", mean=0, std=1)
gen.add_variable("x2", dist="normal", mean=0, std=1)
gen.add_treatment("treat", assignment="random", start_period=20)
gen.set_outcome(formula="y ~ 2*x1 + 0.5*x2 + 3*treat", noise_std=1.0)

# 2. Heterogeneous treatment effect
gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")

# 3. Pathologies
gen.add_structural_break(period=20, target="y", kind="mean_shift", magnitude=2.0)
gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=False)
gen.add_selection_bias(rule="x1 > 1.5", drop_prob=0.4)

# 4. Validate combo (permissive + warnings)
gen.validate_combo()

# 5. Generate
df, truth = gen.generate()

# 6. Stress test a single model
test = StressTest(truth)
report = test.evaluate(fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df)
report.summary()

# 7. Compare multiple models
results = compare_models(
    data=df,
    truth=truth,
    models={
        "OLS": (reference.ols_fit, reference.ols_predict),
        "DiD": (reference.did_fit, reference.did_predict),
        "DoubleML": (reference.doubleml_fit, reference.doubleml_predict),
        "sklearn_LR": (reference.sklearn_lr_fit, reference.sklearn_lr_predict),
    },
)
results.ranked_table(by="default_composite")
results.ranked_table(by="coef_rmse")

# 8. Visualization
plot_recovery_report(report)
plot_recovery_report(results)
```

## Package layout

```
spuriosity/
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── .gitignore
├── docs/
│   └── design_spec.md
├── src/
│   └── spuriosity/
│       ├── __init__.py
│       ├── generator.py       # PanelGenerator
│       ├── pathologies.py     # StructuralBreak, Confounder, SelectionBias, base class
│       ├── ground_truth.py    # GroundTruth dataclass
│       ├── stress_test.py     # StressTest, compare_models
│       ├── reference.py       # batteries-included fit functions
│       └── viz.py             # plot_recovery_report
└── tests/
    ├── test_generator.py
    ├── test_pathologies.py
    └── test_stress_test.py
```

## Deferred to v1.1+

- Unit root pathology
- Multi-dimensional HTE modifiers
- Polars output support
- Cross-version reproducibility guarantees

## v2 additions

### Metrics registration API (`spuriosity.metrics`)

v1 hard-coded its four component metrics (`coef_rmse`, `confounding_bias`,
`break_detection_lag`, `cate_rmse`-as-an-unimplemented-slot) directly inside
`StressTest.evaluate()`. v2 extracts this into a pluggable registry:

- `MetricContext` bundles everything a metric function might need (truth,
  fitted coefficients, raw fit result, data, fit_fn, fit_kwargs, period_col)
  so metric signatures stay uniform regardless of complexity.
- A metric function is `Callable[[MetricContext], float | dict[str, float] | None]`.
  Returning `None` (or an empty dict) means "not applicable," following the
  v1 convention of omitting inapplicable metrics rather than reporting a
  placeholder 0.0/NaN.
- A dict-returning metric produces multiple result keys, namespaced as
  `f"{metric_name}:{key}"`, except a key matching the metric's own
  registered name, which is stored bare. This is how the built-in
  `confounding_bias` metric produces both a bare aggregate and
  `confounding_bias:x1`-style per-feature keys from a single registered
  function.
- `MetricRegistry.copy()` supports starting from
  `spuriosity.metrics.default_registry` (the four built-ins) and adding or
  overriding metrics without mutating the shared default.
- `StressTest(truth, metric_registry=...)` and
  `compare_models(..., metric_registry=...)` both accept a custom registry;
  omitting it uses `default_registry`.

This closes the most-requested v1 gap (external users wanting to test
their own metric against a spuriosity-generated DGP) without requiring
users to fork or monkey-patch `StressTest`.

### Multi-dimensional HTE

v1 shipped single-dimension HTE only (`modifier: str`), deferred
multi-dimensional CATE surfaces to a later release. v2 extends
`PanelGenerator.add_hte(modifier=...)` and `spuriosity.hte.HTE` to accept
either a single column name or a list of column names, while keeping the
single-dimension case's public call signature **exactly unchanged** from
v1 for backward compatibility:

- Single modifier (str, or a length-1 list): `GroundTruth.true_cate` is
  `Callable[[float], float]`, called positionally as `f(x)` -- identical
  to v1.
- Multiple modifiers (list of 2+ names): `GroundTruth.true_cate` instead
  requires keyword arguments matching each modifier's name, e.g.
  `f(x1=1.0, x2=2.0)`, to avoid positional-argument-order ambiguity. A
  positional call raises `TypeError`; a missing/extra keyword raises a
  `TypeError` naming the specific modifier(s) involved.
- `HTE.modifiers` (plural, a list) is always available; `HTE.modifier`
  (singular, a str) is preserved for backward compatibility but raises
  `AttributeError` if the HTE is multi-dimensional, directing callers to
  `.modifiers`.
- `evaluate_on_column(array)` (singular, v1) is preserved for the
  single-dimension case; `evaluate_on_columns({name: array, ...})`
  (plural, v2) is the general-case vectorized evaluator used internally
  by `PanelGenerator` regardless of dimensionality.
- `plot_recovery_report`'s CATE panel (which plots a 1D curve) raises a
  clear `ValueError` if given a multi-dimensional `true_cate` rather than
  crashing on a confusing `TypeError` from the positional call it makes
  internally; multi-dimensional CATE *surface* plotting is not
  implemented. Coefficient-recovery plotting is unaffected either way.

Verified end-to-end with a real binned mean-differencing estimator against
actual generated 2D data (treated-vs-control outcome gap at
`(x1=0, x2=0)` matches `true_cate(x1=0, x2=0)`, and at `(x1=2, x2=1)`
matches `true_cate(x1=2, x2=1)`), and a 3-modifier case confirming the
implementation generalizes beyond the 2D special case.

### Heteroskedasticity pathology

`PanelGenerator.add_heteroskedasticity(feature, formula)` makes the
outcome's noise standard deviation vary with `feature` according to a
`pandas.eval`-evaluated `formula` multiplier (e.g. `"1 + 0.5*x1**2"`),
instead of the constant `noise_std` from `set_outcome`. Multiple calls
compose multiplicatively, consistent with how `StructuralBreak`'s
`variance_shift` kind already composes. Negative multiplier values are
clamped to 0.

This does not bias OLS point estimates but invalidates naive (non-robust)
standard errors -- verified end-to-end: a fitted coefficient stays within
0.05 of the true value at n=100k, while HC3-robust standard errors exceed
naive OLS standard errors by >30%, the textbook signature of
heteroskedasticity going undetected by a naive fit.

### Multicollinearity pathology

`PanelGenerator.add_multicollinearity(feature, correlated_with, correlation)`
generates a **new** column (`feature` must not already be declared -- this
is the opposite convention from `Confounder`/`Heteroskedasticity`, which
modify an *existing* column) as a near-linear function of an existing
`correlated_with` column, calibrated to Pearson correlation `correlation`
via `feature = rho*z + sqrt(1-rho^2)*epsilon` on the standardized parent
variable. `correlation` must be in `[0, 1)`; exactly 1.0 is disallowed
since perfect collinearity makes OLS undefined rather than merely
high-variance.

The implied VIF for the collinear pair is the closed-form
`1 / (1 - rho**2)`; verified against a real `statsmodels`
`variance_inflation_factor` computation on generated data (predicted 5.26
vs. actual 5.22 at rho=0.9, n=100k).

Name-collision checking is bidirectional: declaring a variable with a name
already claimed by an earlier `add_multicollinearity` call raises, and
vice versa -- `_check_name_available` was extended to include
multicollinearity-generated names in its taken-name set, closing a gap
that existed only in one direction before this was added.

### Measurement error pathology

`PanelGenerator.add_measurement_error(feature, noise_std)` injects
classical errors-in-variables into `feature` (which must already be
declared): the outcome is generated from the TRUE (pre-error) values, but
the `feature` column in the final DataFrame is replaced with
`true_value + N(0, noise_std**2)`. This is the mechanistic mirror image of
`Confounder` -- `Confounder` modifies a feature *before* the outcome reads
it (so the corruption affects the true relationship); `MeasurementError`
modifies it *after* (so only what's observable is corrupted, and the
outcome's real dependence is on a value the researcher never actually
sees).

The classical result this pathology exists to let users verify: a naive
regression of the outcome on the noisy *observed* feature has its
coefficient attenuated toward zero by the reliability ratio
`Var(true) / (Var(true) + noise_std**2)` -- the opposite direction of bias
from confounding (which inflates a coefficient away from zero).
`GroundTruth.measurement_error[i].reliability_ratio` reports the
*realized* ratio (computed from the true values' actual sample variance,
not a theoretical population value), so it reflects the specific generated
dataset. Verified end-to-end: naive OLS on the corrupted feature matches
the predicted attenuated coefficient to within 0.02 at n=1M, and a direct
residual check confirms the outcome was built from the true pre-noise
values (not the corrupted ones) by showing residuals computed using the
*observed* feature retain variance on the order of the injected
measurement noise itself, rather than the outcome's own (much smaller)
noise term.
