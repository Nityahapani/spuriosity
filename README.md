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
  DiD, DoubleML, logit) so a first stress test runs in ~5 lines.
- `compare_models()` — benchmark multiple models against the same DGP with a
  transparent, user-overridable composite ranking plus always-visible individual
  metrics.
- `plot_recovery_report()` — visualize coefficient recovery, power curves, and
  CATE estimates vs. ground truth.

## Installation

Not yet published to PyPI. Once available:

```bash
pip install spuriosity
```

## Quickstart (target v1 API)

```python
from spuriosity import PanelGenerator, StressTest, reference

gen = PanelGenerator(n_entities=500, n_periods=40, seed=42)
gen.add_variable("x1", dist="normal", mean=0, std=1)
gen.add_treatment("treat", assignment="random", start_period=20)
gen.set_outcome(formula="y ~ 2*x1 + 3*treat", noise_std=1.0)
gen.add_confounder(feature="x1", outcome="y", strength=0.6, observed=False)

df, truth = gen.generate()

test = StressTest(truth)
report = test.evaluate(fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df)
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
