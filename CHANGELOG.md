# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); this project
follows [Semantic Versioning](https://semver.org/) from `v0.2.0` onward.

## [0.2.0] — 2026-07-19

The v2 release. Adds a full second wave of injectable pathologies, a
pluggable metrics API, and a much larger reference-fit toolbox spanning
classical econometrics through modern causal ML — while remaining fully
backward compatible with `v0.1.0`'s public API (verified: every v0.1.0
example and test still passes unchanged).

### Added

**Metrics & evaluation**
- `spuriosity.metrics` module: `MetricContext`, `MetricRegistry`,
  `default_registry` — pluggable metric registration, closing the
  most-requested v1 gap (users can register their own
  `metric_fn(ctx) -> float | dict[str, float] | None` without forking
  `StressTest`). `StressTest` and `compare_models` both accept a custom
  `metric_registry=`.

**New pathologies** (`PanelGenerator.add_*`)
- `add_heteroskedasticity(feature, formula)` — non-constant error
  variance; verified via naive-vs-robust standard error divergence.
- `add_multicollinearity(feature, correlated_with, correlation)` —
  correlated regressors; verified via a closed-form VIF match against
  `statsmodels`.
- `add_measurement_error(feature, noise_std)` — classical
  errors-in-variables; verified via the closed-form attenuation-bias
  (reliability ratio) formula.
- `add_endogeneity(feature, instrument, instrument_strength,
  endogeneity_strength)` — regressor correlated with the outcome's error
  term, with a paired exogenous instrument; verified via naive-OLS bias
  and 2SLS recovery, including a deliberately weak-instrument case.
- `add_unit_root(feature, drift=0.0)` — random-walk nonstationarity;
  verified via the Augmented Dickey-Fuller test and a real reproduction
  of the classic Granger-Newbold (1974) spurious-regression result
  (~76% false-positive rate for independent random walks vs. ~6% for
  i.i.d. series, across 200 simulated datasets).
- `add_treatment(assignment="propensity", propensity_formula=...)` —
  covariate-dependent binary treatment assignment, closing a gap that
  blocked clean PSM and panel FE/RE test scenarios.
- Multi-dimensional HTE: `add_hte(modifier=["x1", "x2", ...])`, extending
  v1's single-dimension-only support. `GroundTruth.true_cate` requires
  keyword arguments for multi-dim HTE (`true_cate(x1=..., x2=...)`);
  single-dimension HTE keeps its original positional signature unchanged.

**New `GroundTruth` fields**: `heteroskedasticity`, `multicollinearity`,
`measurement_error`, `endogeneity`, `unit_root` (all default to empty
lists — purely additive, no existing field changed shape).

**New reference fits** (`spuriosity.reference`)
- `iv2sls_fit` / `iv2sls_predict` + `first_stage_f_stat` — 2SLS via
  `linearmodels`, paired with `add_endogeneity`.
- `panel_fe_fit` / `panel_fe_predict`, `panel_re_fit` / `panel_re_predict`
  — panel fixed/random effects via `linearmodels`.
- `hausman_test(fe_result, re_result)` — the standard FE-vs-RE
  specification test, using eigenvalue-clipped pseudo-inversion for
  numerical robustness (the naive formula can produce a nonsensical
  chi-squared statistic at realistic finite sample sizes).
- `psm_fit` / `psm_predict` — nearest-neighbor propensity score matching,
  with a common-support overlap diagnostic.
- `xgboost_fit` / `xgboost_predict` — a pure predictive ML baseline
  (`.coefficients` deliberately empty; `.extra["feature_importances"]`
  instead).
- `causal_forest_fit` / `causal_forest_predict` — `econml`'s
  `CausalForestDML`, for full CATE-surface recovery rather than a single
  ATE. `causal_forest_predict` returns per-row CATE, not an outcome
  prediction (documented as breaking the usual `*_predict` convention).

**Synthetic control method** (`spuriosity.synthetic_control`)
- `synthetic_control_fit` / `SyntheticControlResult` — the full
  Abadie-Diamond-Hainmueller method with placebo-in-space inference (kept
  full scope per project decision, not descoped). Documents a genuine,
  literature-known limitation of placebo-in-space with small donor pools,
  found during test development.

**Docs**
- `docs/API.md` — a unified API reference, verified against the
  installed package by `scripts/verify_api_doc.py` rather than written
  from memory. Every documented symbol, signature, and standalone code
  example is checked to actually exist/run as documented.
- `examples/` — three worked notebooks (DiD under selection bias, causal
  forest vs. OLS on a confounded + heterogeneous DGP, IV under weak
  instruments), each executed end-to-end against a genuinely blank Python
  environment before being committed.
- `docs/v3_planning.md` — the v3 roadmap.

**New optional dependency groups**: `linearmodels`, `xgboost`, `econml`
(alongside the existing `sklearn`, `viz`, `doubleml`).

**New top-level exports**: `Heteroskedasticity`, `Multicollinearity`,
`MeasurementError`, `Endogeneity`, `UnitRoot` (pathology classes) and
their `GroundTruth` info records (`HeteroskedasticityInfo`,
`MulticollinearityInfo`, `MeasurementErrorInfo`, `EndogeneityInfo`,
`UnitRootInfo`); `MetricContext`, `MetricRegistry`, `default_registry`;
`synthetic_control_fit`, `SyntheticControlResult`. Also closed a v0.1.0
inconsistency by exporting `StructuralBreak`, `Confounder`,
`SelectionBias`, and `Pathology` at the top level (previously only
reachable via `spuriosity.pathologies.*`), matching `HTE`'s existing
top-level export.

### Changed

- `StressTest.evaluate()`'s four built-in metrics (`coef_rmse`,
  `confounding_bias`, `break_detection_lag`, `cate_rmse`) are now
  implemented via the new `MetricRegistry` rather than hard-coded inline
  — behavior is unchanged (verified byte-for-byte equivalent against the
  pre-refactor implementation), only the extension mechanism is new.
- `FitResult` gained an `extra: dict[str, Any]` field for
  fit-function-specific metadata (used by `iv2sls_fit`'s column-layout
  bookkeeping, `psm_fit`'s propensity scores/common support,
  `xgboost_fit`'s feature importances).

### Fixed

- A real order-dependence bug in `RNGManager`: dynamic stream names were
  originally assigned spawn-tree slots by request order, meaning the same
  `PanelGenerator` spec could silently produce different data depending
  on the order `.add_*()` calls were made. This was actually a `v0.1.0`
  regression risk caught and fixed during `v0.2.0` metric-registry work
  when the underlying `PanelGenerator` internals were touched again.

### Known limitations carried forward (see `docs/v3_planning.md`)

- `cate_rmse` remains an unimplemented metric slot (always returns
  `None`) — no established convention yet for row-level CATE prediction
  comparison.
- `iv2sls_fit`/`panel_re_fit` report their intercept under
  `linearmodels`' `"const"` key rather than patsy's `"Intercept"`
  convention used elsewhere.
- `sklearn_lr_fit` does not guard its `sklearn` import with a clear
  `ImportError` message, unlike every other optional-dependency function.
- No dedicated pathology for a time-invariant panel entity random effect
  (the classic FE-vs-RE test scenario still requires manual construction).

## [0.1.0] — 2026-07-19

Initial release. Full v1 feature set:

- `PanelGenerator` — panel data generation with declared variables,
  treatment, and a patsy-formula or callable outcome DGP.
- Three pathologies: `StructuralBreak` (mean/variance/coefficient shift),
  `Confounder` (with closed-form omitted-variable-bias prediction),
  `SelectionBias` (via a constrained `pandas.eval`).
- Single-dimension `add_hte` for heterogeneous treatment effects.
- `spuriosity.reference` — five reference fits: `ols_fit`,
  `sklearn_lr_fit`, `did_fit`, `doubleml_fit`, `logit_fit`.
- `StressTest` / `compare_models` — model evaluation against known ground
  truth, with a transparent, user-overridable composite ranking.
- `plot_recovery_report` — coefficient recovery and model-comparison
  visualization.
- `GroundTruth` — frozen dataclass exposing the true DGP alongside every
  generated dataset, with `.to_dict()`/`.to_json()` serialization.
