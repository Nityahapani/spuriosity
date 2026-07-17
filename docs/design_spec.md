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

### Endogeneity/IV pathology + IV/2SLS reference fit

`PanelGenerator.add_endogeneity(feature, instrument, instrument_strength, endogeneity_strength, first_stage_noise_std=0.5)`
makes `feature` endogenous (correlated with the outcome's error term via a
shared latent variable `u`) and generates a new, exogenous `instrument`
column, mirroring the standard textbook IV construction:

```
instrument ~ N(0, 1)                                     [new column]
u ~ N(0, 1)                                                [latent]
feature = instrument_strength*instrument + endogeneity_strength*u + noise
outcome_mean += endogeneity_strength*u
```

`feature` must already be declared (its values are replaced, matching
`Confounder`'s convention); `instrument` must NOT already be declared (it
is created fresh, matching `Multicollinearity`'s convention). Both
directions of name-collision checking apply, same bidirectional pattern
established for `Multicollinearity`.

`instrument_strength` set low relative to `endogeneity_strength`
generates a *weak* instrument on purpose -- this is the more interesting
stress-test case than "IV always works." `GroundTruth.endogeneity[i]`
exposes `realized_first_stage_f_stat` (the standard weak-instrument
diagnostic, computed on the actual generated sample via a hand-rolled OLS
F-test verified to match `statsmodels`' `.fvalue` to 6 decimal places),
so users can check whether their generated dataset crosses the classic
Stock-Yogo rule-of-thumb threshold of 10.

Paired reference fit: `spuriosity.reference.iv2sls_fit` /
`iv2sls_predict`, using `linearmodels.iv.IV2SLS` (new optional dependency,
`pip install spuriosity[linearmodels]`, lazy-imported with a clear
`ImportError` if missing, following the same pattern as `doubleml_fit`).
Also ships `reference.first_stage_f_stat(data, endogenous, instruments)`,
a `statsmodels`-only convenience (no `linearmodels` required) for
diagnosing instrument strength independent of fitting a full 2SLS model.

Note: `linearmodels` uses `"const"` as its intercept column name, while
`GroundTruth.true_coefficients` and the rest of `spuriosity.reference`
use patsy/statsmodels' `"Intercept"` convention. These are not currently
reconciled automatically -- `coef_rmse` still works correctly since it
only compares keys present in both dicts (so the differently-named
intercept term is simply excluded from that particular comparison, not a
source of a wrong value), but this is a known rough edge worth revisiting
if `linearmodels`-based fits become more central to the toolbox.

Verified end-to-end through the real API: naive OLS on the endogenous
feature is meaningfully biased (>0.3 off truth at n=500k); 2SLS via
`iv2sls_fit` recovers the true coefficient to within 0.02; a deliberately
weak instrument (`instrument_strength=0.01`) produces a realized F-stat
of ~2.5 (well below the weak-instrument threshold) and a 2SLS estimate
with standard error an order of magnitude larger than under a strong
instrument -- confirming the pathology genuinely produces the "IV
strategy becomes unreliable" failure mode, not just "IV works when
everything is textbook-strong."

### Unit root pathology

`PanelGenerator.add_unit_root(feature, drift=0.0)` converts `feature` from
i.i.d. draws into a random walk (with optional drift), reusing the same
underlying values `add_variable` already drew as increments rather than
generating fresh noise. Unlike every other pathology in v2, this one
operates on the full panel structure (`UnitRoot.apply_to_panel(df)` takes
the whole DataFrame, not a flat array), since the cumulative sum must
reset independently at each entity boundary -- verified with a dedicated
row-order-preservation test in addition to the basic per-entity cumsum
correctness check.

Two verified statistical properties:

1. **Formal nonstationarity**: the Augmented Dickey-Fuller test fails to
   reject the unit-root null for a generated random-walk series
   (p > 0.1), while correctly rejecting it for an otherwise-identical
   i.i.d. series (p < 0.05) -- both checked via `statsmodels.tsa.stattools.adfuller`
   on real generated data, not a synthesized array.

2. **Spurious regression** (the Granger-Newbold 1974 result): OLS between
   two *independent* unit-root series shows a "significant" coefficient
   (p < 0.05) far more often than the nominal significance level implies.
   Verified end-to-end through the real `PanelGenerator` API across many
   independent simulated datasets: ~76% false-positive rate for
   independent random walks vs. ~6% for independent i.i.d. series (close
   to the nominal 5%), reproduced as a permanent test
   (`test_spurious_regression_false_positive_rate_inflated`) rather than
   a one-off validation script. This is the most striking and
   pedagogically important result this pathology reproduces -- it's the
   textbook demonstration of why nonstationary time series need
   differencing/cointegration analysis rather than naive OLS.

### Panel Fixed Effects / Random Effects + Hausman test

`spuriosity.reference.panel_fe_fit` / `panel_fe_predict` and
`panel_re_fit` / `panel_re_predict` (new optional `linearmodels` usage,
same dependency already introduced for `iv2sls_fit`) wrap
`linearmodels.panel.PanelOLS(entity_effects=True)` and
`linearmodels.panel.RandomEffects` respectively, the workhorse pairing of
applied panel econometrics.

- **FE** demeans within each entity, controlling for all time-invariant
  entity-level confounders (observed or not) at the cost of being unable
  to estimate any time-invariant regressor's coefficient and having no
  intercept term.
- **RE** treats entity effects as a random draw uncorrelated with the
  regressors -- more efficient than FE when that assumption holds, but
  biased when it doesn't.

`spuriosity.reference.hausman_test(fe_result, re_result)` implements the
standard specification test: compares only the coefficients shared
between both fits (RE's intercept, which has no FE counterpart, is
excluded automatically) and tests the null that entity effects are
uncorrelated with the regressors. A small p-value means RE is biased and
FE should be preferred.

The classical Hausman formula assumes `Var(FE) - Var(RE)` is positive
semi-definite, but finite-sample covariance estimates can violate this
slightly due to estimation noise (confirmed during development: even at
n=2000×10, a naive `np.linalg.inv` on the raw covariance difference
produced a small negative eigenvalue, making the naive chi-squared
statistic nonsensical). Fixed via eigenvalue-clipped pseudo-inversion --
the standard practical workaround used by applied econometrics software
-- verified against both directions: correctly rejects RE (p≈0) when an
entity effect is constructed to correlate with the regressor, and
correctly fails to reject (p=0.26-0.76 across test runs) when the entity
effect is independent of the regressor.

Note: `spuriosity` does not yet have a dedicated pathology for injecting
a time-invariant panel entity effect (the classic FE-vs-RE test scenario
had to be constructed by hand in tests, post-hoc-modifying a generated
DataFrame rather than through a first-class `add_*` builder method) --
flagged as a natural candidate for a future pathology rather than
addressed here, to avoid scope creep into a Tier 1 item that was
specifically about the reference-fit toolbox, not new DGP surface.

Verified end-to-end via `compare_models`: on data with an entity effect
correlated with the regressor, FE (coef_rmse ≈ 0.0004) dramatically
outranks both RE (≈ 0.146) and pooled OLS (≈ 0.210), reproducing the
textbook demonstration of why panel FE is the default choice when entity
heterogeneity might correlate with regressors of interest.

### Propensity score matching reference fit

`spuriosity.reference.psm_fit` / `psm_predict` (uses the `sklearn`
optional dependency already introduced for `sklearn_lr_fit`) estimates
the average treatment effect on the treated (ATT) for a binary treatment
via nearest-neighbor propensity score matching:

1. Fit a logistic regression of `treatment` on `covariates` to estimate
   each unit's propensity score.
2. Match each treated unit to its nearest control unit by propensity
   score (nearest-neighbor, with replacement, via `scipy.spatial.cKDTree`
   -- scipy is already a core dependency, so no new dependency needed for
   the matching step itself).
3. Average the matched-pair outcome differences.

`.extra["common_support_fraction"]` reports the fraction of treated units
whose propensity score falls within the control group's observed
propensity range -- a standard overlap diagnostic. Verified to correctly
distinguish a well-overlapping scenario (0.9999 at moderate propensity
separation) from a poorly-overlapping one (0.941 at extreme separation),
confirming it's a real signal and not just a placeholder value.

Verified end-to-end against a hand-constructed confounded binary
treatment on top of `spuriosity`-generated covariate data (see the note
under Panel FE/RE about `spuriosity` not yet having a dedicated
covariate-dependent binary treatment pathology -- `add_treatment`'s
`"random"` assignment is deliberately independent of covariates, so this
scenario has to be built manually, same limitation encountered twice now):
naive difference-in-means is meaningfully biased (>0.3 off the true ATE
at n=100k), while `psm_fit` recovers the true ATE to within 0.1.

### Propensity-based treatment assignment (closes a recurring gap)

`PanelGenerator.add_treatment(assignment="propensity", propensity_formula=...)`
extends treatment assignment beyond `"random"` (independent of
covariates) to support genuine covariate-dependent selection into
treatment: each entity's treatment probability is
`sigmoid(propensity_formula)`, evaluated once per entity using that
entity's covariate values at period 0 (treatment remains entity-fixed and
time-invariant before `start_period`, matching the existing `"random"`
semantics). `propensity_formula` uses the same `pandas.eval` mechanism
and security posture as `add_selection_bias`/`add_hte` (verified: code
injection attempts are rejected the same way).

This closes a gap flagged three times across the panel FE/RE, PSM
work: those reference-fit test scenarios needed a *confounded* binary
treatment (propensity correlated with an observed covariate) to be
meaningful test cases, and previously had to construct this by hand,
post-hoc-modifying a generated DataFrame rather than using a first-class
`PanelGenerator` API. `tests/test_psm.py` has been refactored to use
`assignment="propensity"` directly; verified byte-for-byte equivalent
statistical results to the old hand-constructed version (naive
diff-in-means bias >0.3, PSM recovery within 0.1 of true ATE at n=100k).

Verified independently: empirical treatment rate within a narrow
covariate window matches the theoretical `sigmoid(propensity_formula)`
value to within 0.05 at three test points (x1 = -1, 0, 1), confirming the
mechanism produces genuinely calibrated propensity-based assignment, not
just "treated units differ from control on average."

Note: this does NOT close the separate gap noted under Panel FE/RE (a
time-invariant per-entity *random effect* added to a continuous
covariate/outcome, for the classic entity-effect-correlated-with-regressor
FE-vs-RE test scenario) -- that remains a distinct, still-open follow-up,
since it's a different mechanism (a continuous entity-level nuisance
term, not a binary treatment's selection probability).

### XGBoost ML baseline

`spuriosity.reference.xgboost_fit` / `xgboost_predict` (new optional
`xgboost` dependency, `pip install spuriosity[xgboost]`, lazy-imported
with a clear `ImportError` if missing) is a pure predictive baseline for
the "does a flexible nonlinear model beat econometric assumptions"
comparison -- e.g. against `ols_fit` on data generated with a nonlinear
outcome (`set_outcome(fn=...)`).

Unlike every other reference fit, `.coefficients` is deliberately left
empty: XGBoost produces no interpretable per-feature coefficient, so
`coef_rmse` is simply not computed for it (the same "not applicable"
convention used throughout `StressTest`/`compare_models` -- verified that
`ranked_table(by="coef_rmse")` correctly excludes an XGBoost model while
still including a comparable OLS model in the same comparison, tracked
via `.attrs["excluded_models"]` rather than silently dropped).
`.extra["feature_importances"]` exposes XGBoost's own importance metric
instead, explicitly documented as a different quantity that should not be
compared against `GroundTruth.true_coefficients`.

Verified in both directions: on genuinely nonlinear generated data,
XGBoost's R² substantially exceeds OLS's (0.986 vs 0.608 in one test
run); on genuinely linear generated data, XGBoost shows no meaningful
predictive advantage over correctly-specified OLS (confirming the
nonlinear-data win reflects real functional-form flexibility rather than
XGBoost being unconditionally "better").
