# spuriosity

Synthetic panel and time-series data with **known ground truth** and deliberately
injectable econometric pathologies — for stress-testing ML pipelines and
benchmarking causal inference methods.

Most synthetic data tools (SDV, Gretel, Synthea) optimize for *realism*: matching a
real dataset's distribution for privacy-safe sharing or prototyping. `spuriosity`
optimizes for the opposite. You define the true data-generating process, then
deliberately inject known failure modes drawn from econometrics — structural
breaks, latent confounding, selection bias — and every generated dataset ships
with the ground truth (true coefficients, true treatment effects, true CATE
function). That lets you directly measure whether a model or estimator recovers
the truth, or gets fooled by a specific, named pathology.

> **Status**: early alpha, under active design/development. API surface described
> below is the target for v1 and may still shift. See
> [`docs/design_spec.md`](docs/design_spec.md) for the full design rationale.

## Why

- Realistic-looking synthetic data (SDV, Gretel) has no ground truth to check
  against — you can't tell whether your pipeline recovered the *right* answer,
  only how it performed on a holdout set.
- Robustness pathologies used in causal inference and ML research (structural
  breaks, confounding, selection bias) currently live as one-off scripts buried
  inside individual papers' codebases — not as a general, composable,
  pip-installable toolkit.
- `spuriosity` packages the econometrics toolkit for *known, controllable
  failure modes* into a reusable API, with ground truth and built-in evaluation.

## Planned v1 feature set

- `PanelGenerator` — build a panel dataset (n_entities × n_periods; `n_entities=1`
  covers the pure time-series case) with a known DGP, using `patsy` formulas or
  Python callables.
- Injectable pathologies: **structural break**, **confounding**, **selection bias**.
- Single-dimension heterogeneous treatment effects (HTE), with a true CATE
  function exposed in the ground truth.
- `StressTest` — evaluate any model (function-based `fit_fn`/`predict_fn`)
  against the ground truth.
- `spuriosity.reference` — batteries-included fits (OLS, sklearn LinearRegression,
  DiD, DoubleML, logit) so a first stress test runs in a few lines.
- `compare_models()` — benchmark multiple models against the same DGP with a
  transparent, user-overridable composite ranking plus always-visible individual
  metrics.
- `plot_recovery_report()` — visualize coefficient recovery, power curves, and
  CATE estimates vs. ground truth.

## Installation

`spuriosity` is not yet published to PyPI. Install from source (editable,
recommended for development):

```bash
git clone https://github.com/Nityahapani/spuriosity.git
cd spuriosity
pip install -e ".[dev,viz]"
```

That pulls in everything: `numpy`, `pandas`, `patsy`, `statsmodels`, `scipy`,
plus the `viz` extras (matplotlib), `sklearn` and `doubleml` for the reference
fits, and `pytest`/`ruff`/`mypy` for development.

For a minimal install without the optional extras:

```bash
pip install -e .
```

## Quickstart

The `set_outcome` formula is **right-hand side only** (the outcome column
`y` is being generated, so it can't appear on the LHS of its own DGP). True
coefficients are supplied as a separate dict, keyed by the resulting
design-matrix column names (including `"Intercept"`, which patsy adds
automatically).

```python
from spuriosity import PanelGenerator, StressTest, reference

gen = PanelGenerator(n_entities=500, n_periods=40, seed=42)
gen.add_variable("x1", dist="normal", mean=0, std=1)
gen.add_treatment("treat", assignment="random", start_period=20)
gen.set_outcome(
    formula="x1 + treat",  # RHS only — `y` is the generated outcome
    coefficients={"x1": 2.0, "treat": 3.0, "Intercept": 0.0},
    noise_std=1.0,
)
gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=False)

df, truth = gen.generate()

test = StressTest(truth)
report = test.evaluate(
    fit_fn=reference.ols_fit,
    predict_fn=reference.ols_predict,
    data=df,
    fit_kwargs={"formula": "y ~ x1 + treat"},  # full patsy here (LHS + RHS)
    model_name="OLS",
)
report.summary()
```

See [`docs/design_spec.md`](docs/design_spec.md) for the full API design and
architectural decisions.

## Reproducibility

Same `seed` + same pinned `spuriosity` and `numpy` versions produces a
byte-identical dataset. Cross-version reproducibility is **not** guaranteed —
pin and cite both versions for reproducible research.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
