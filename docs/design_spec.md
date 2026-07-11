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
