# spuriosity v3 planning

Following the completion of v2 Tier 1 (13/13 items shipped, 419 tests
passing — see `docs/design_spec.md` for the full v2 changelog and
`docs/API.md` for the current, verified API surface), this issue tracks
what's next.

Items are grouped by how they were identified: **explicit deferrals**
(flagged when a v2 decision was made), **real gaps found during v2
implementation** (discovered while building something else, not planned
up front), and **the original v2 Tier 2/3 backlog** (carried forward,
still not started).

---

## Tier 1 — real gaps found during v2 (highest priority; each blocked something already)

These aren't speculative — each one already caused friction or a
workaround during v2 and is likely to recur.

### 1. Panel entity random-effect pathology

`spuriosity` has no dedicated pathology for injecting a time-invariant
per-entity random effect on a continuous covariate/outcome. This
specifically blocked writing a clean `panel_fe_fit` vs. `panel_re_fit`
test/example scenario (the classic "entity effect correlated with
regressor → RE biased, FE consistent" demonstration) — it had to be
constructed by hand, post-hoc-modifying a generated DataFrame, in both
`tests/test_panel_fe_re.py` and `docs/API.md`'s reference-fit example.

Proposed API: `add_entity_effect(feature_or_outcome, strength, correlated_with=None)`,
mirroring `add_confounder`'s "modifies an existing column, contributes to
the outcome" pattern but drawn once per entity (like treatment
assignment) rather than once per row.

Note: this is a *different* mechanism from `add_treatment(assignment="propensity")`
(shipped in v2, closes covariate-dependent **binary treatment**
assignment) — don't conflate the two when scoping this.

### 2. `cate_rmse` metric — still just a slot

`spuriosity.metrics.default_registry` registers `cate_rmse` but
`_cate_rmse()` always returns `None` — it's a placeholder carried over
unimplemented since v1. Blocked by an unresolved design question: there's
no established convention for how a `fit_fn`/`FitResult` should expose a
*row-level* CATE prediction for comparison against `GroundTruth.true_cate`.
(`causal_forest_predict` does return per-row CATE, breaking the usual
predict-returns-outcome convention specifically to support this — but
`cate_rmse` doesn't yet make use of it.)

Needs: decide the convention (e.g. `FitResult.extra["cate_predict_fn"]`,
or a documented special case in `MetricContext`), implement `_cate_rmse`
for real, add it to the `causal_forest_fit`/HTE test suite.

### 3. `linearmodels` "const" vs. patsy "Intercept" naming mismatch

`iv2sls_fit` and `panel_re_fit` report their intercept under
`linearmodels`' own `"const"` key; every other reference fit uses
patsy/statsmodels' `"Intercept"` convention (matching
`GroundTruth.true_coefficients`). `coef_rmse` still works correctly today
since it only diffs shared keys — but this means the intercept is
silently excluded from that comparison for these two fits, which could
surprise a user who doesn't know to look for it. Worth either (a)
normalizing `"const"` → `"Intercept"` in `FitResult.coefficients` for
these two fits, or (b) at minimum surfacing a warning when `coef_rmse` is
computed against a `FitResult` known to use the `"const"` convention.

### 4. `sklearn_lr_fit` doesn't guard its `sklearn` import

Every other optional-dependency function (`doubleml_fit`, `iv2sls_fit`,
`psm_fit`, `xgboost_fit`, `causal_forest_fit`, `panel_fe_fit`,
`panel_re_fit`) wraps its import in a `try/except ImportError` that
raises a clear `spuriosity`-authored message pointing at the right `pip
install spuriosity[extra]` command. `sklearn_lr_fit` imports
`sklearn.linear_model.LinearRegression` directly — if `scikit-learn`
isn't installed, the user gets a raw `ModuleNotFoundError` instead.
Trivial fix, just needs doing (found while writing `docs/API.md`, footnoted
there rather than fixed, to avoid scope creep into a docs-only commit).

---

## Tier 2 — explicit deferrals from v1/v2 design decisions

### 5. Cross-version reproducibility

Currently: same `seed` + same pinned `spuriosity`/`numpy` versions →
byte-identical output; no guarantee *across* versions. Worth deciding
whether this is ever worth pursuing (likely answer: no, chasing
numpy/scipy's own RNG stability guarantees is a maintenance trap — but
worth an explicit decision + doc update rather than leaving it as
perpetually "not yet").

### 6. Polars output support

Flagged as a v1.1+ deferral, never revisited. Worth a real cost/benefit
pass: how much of the codebase assumes `pandas.DataFrame` specifically
(patsy's `dmatrix`, `pandas.eval` for `add_selection_bias`/`add_hte`,
`.groupby` for `UnitRoot`) vs. how much could be backend-agnostic.
Likely a bigger lift than it sounds; scope carefully before committing.

---

## Tier 3 — original v2 backlog, not yet started

Carried forward from the initial v2 planning conversation, unchanged in
priority ordering relative to each other, but now *after* Tiers 1–2 above
given what was actually learned building v2.

### 7. Power analysis helpers

Given a DGP, how many observations are needed to detect a true effect of
size X with 80% power? Trivial in principle since `spuriosity` already
has ground truth — mostly a matter of wrapping repeated `generate()` +
`fit` + power calculation into a clean API
(`spuriosity.power_analysis(generator_fn, effect_sizes, n_range, n_sims)`).

### 8. Calibration plots

For CIs: nominal vs. empirical coverage. For point predictions: predicted
vs. actual. Natural extension of `plot_recovery_report` — possibly
`plot_calibration_report(reports: list[StressTestReport])` run across
many `generate()` + fit repetitions at a fixed DGP.

### 9. `spuriosity.diagnose(model, dgp)` — automated diagnostics mode

Runs a model many times against repeated draws from the same DGP spec,
reports power / Type I error / coverage in one call. Effectively
productizes items 7–8 into a single entry point. Should probably be
built *after* 7 and 8 exist as standalone pieces, not before.

### 10. `spuriosity.benchmark` — curated standard DGP testbed

8–12 curated DGPs as a citable standard testbed for new causal inference
method papers (`docs/benchmark.md` + `from spuriosity.benchmark import *`).
This is the highest-leverage, highest-effort item on this whole list —
the "every new causal ML paper cites this" positioning play from the
original v2 plan. Needs real curation work (which DGPs are canonical
enough to include) more than engineering work. Consider drafting the
`docs/benchmark.md` proposal *before* writing any code, and possibly
soliciting outside input given the target audience is external
researchers, not just `spuriosity`'s own maintainer(s).

---

## Non-feature work (carried forward from v2 planning, still relevant)

These were listed as "v2 non-feature work" in the original plan and
mostly didn't happen during v2 Tier 1 (which focused entirely on
features). Worth doing before v3 feature work starts, not after:

- **API freeze** — `docs/API.md` now exists and is self-verifying
  (`scripts/verify_api_doc.py`); the natural next step is deciding *which*
  version this API freezes at (`v0.2.0`?) and adopting real semver from
  that point forward.
- **mypy strict mode** — currently passing under default mypy settings,
  not `--strict`. Worth checking how big the gap is before committing to
  it.
- **GitHub Actions CI** — `pytest` + `ruff` + `mypy` on every PR. Still
  manual (run locally before every commit) as of this issue. The
  pre-flight checklist this project has followed throughout v2 (full
  suite + lint + type check before every push) becomes actually
  *enforced* rather than just habitually followed, once this exists.
- **`CITATION.cff`** — rolled forward from the v0.1.0 backlog, still not
  done.
- **Deprecation policy** — what's removed in v3, what's considered
  stable vs. experimental. Should probably be written *alongside* the API
  freeze decision above, not separately.
- **`CHANGELOG.md`** — semver entries from v0.1.0 onward. Currently the
  only changelog is `git log` + `docs/design_spec.md`'s running
  narrative, which is thorough but not the standard format tooling (and
  users) expect.

---

## Suggested sequencing

1. Tier 1 items (1–4) — small, each closes a real friction point already
   hit twice or more
2. Non-feature work — CI first specifically, since it starts protecting
   every commit from here on rather than just the ones after it's done
3. API freeze decision + `CHANGELOG.md` + `CITATION.cff` together, as one
   "v0.2.0 release" pass
4. Tier 2 items (5–6) — each needs a real scoping decision before any
   code, not just implementation
5. Tier 3 items (7–10) — in order, since 9 depends on 7+8 and 10 is the
   biggest standalone effort on the list
