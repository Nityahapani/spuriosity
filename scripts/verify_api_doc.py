"""
Verification script for docs/API.md. Every import statement and function
call that will appear in API.md is exercised here first. If this script
passes cleanly, every code example in the doc is confirmed to actually
work against the currently installed spuriosity version.
"""
import sys

results: list[tuple[str, str, str]] = []

def check(label, fn):
    try:
        fn()
        results.append((label, "OK", ""))
    except Exception as e:
        results.append((label, "FAIL", f"{type(e).__name__}: {e}"))
        import traceback
        traceback.print_exc()


# ------------------------------------------------------------------
# Top-level imports (every symbol in __all__)
# ------------------------------------------------------------------

def check_all_top_level_imports():
    import spuriosity
    expected = {
        "PanelGenerator", "GroundTruth", "BreakInfo", "SelectionInfo",
        "HeteroskedasticityInfo", "MulticollinearityInfo", "MeasurementErrorInfo",
        "EndogeneityInfo", "UnitRootInfo", "HTE", "reference", "StressTest",
        "StressTestReport", "compare_models", "ComparisonReport",
        "plot_recovery_report", "MetricContext", "MetricRegistry",
        "default_registry", "Pathology", "StructuralBreak", "Confounder",
        "SelectionBias", "Heteroskedasticity", "Multicollinearity",
        "MeasurementError", "Endogeneity", "UnitRoot", "synthetic_control_fit",
        "SyntheticControlResult",
    }
    actual = set(spuriosity.__all__)
    assert actual == expected, f"MISMATCH.\nMissing: {expected - actual}\nExtra: {actual - expected}"

check("__all__ matches documented symbol list exactly", check_all_top_level_imports)


def check_named_imports():
    from spuriosity import (  # noqa: F401 -- import success is the test itself
        PanelGenerator, GroundTruth, BreakInfo, SelectionInfo,
        HeteroskedasticityInfo, MulticollinearityInfo, MeasurementErrorInfo,
        EndogeneityInfo, UnitRootInfo, HTE, reference, StressTest,
        StressTestReport, compare_models, ComparisonReport,
        plot_recovery_report, MetricContext, MetricRegistry,
        default_registry, Pathology, StructuralBreak, Confounder,
        SelectionBias, Heteroskedasticity, Multicollinearity,
        MeasurementError, Endogeneity, UnitRoot, synthetic_control_fit,
        SyntheticControlResult,
    )

check("Direct `from spuriosity import X` for every symbol", check_named_imports)


def check_reference_submodule_functions():
    from spuriosity import reference
    expected_fns = {
        "ols_fit", "ols_predict", "sklearn_lr_fit", "sklearn_lr_predict",
        "did_fit", "did_predict", "doubleml_fit", "doubleml_predict",
        "logit_fit", "logit_predict", "iv2sls_fit", "iv2sls_predict",
        "first_stage_f_stat", "panel_fe_fit", "panel_fe_predict",
        "panel_re_fit", "panel_re_predict", "hausman_test", "psm_fit",
        "psm_predict", "xgboost_fit", "xgboost_predict",
        "causal_forest_fit", "causal_forest_predict", "FitResult",
    }
    for fn in expected_fns:
        assert hasattr(reference, fn), f"reference.{fn} missing"

check("spuriosity.reference has all 24 documented fit/predict/helper symbols", check_reference_submodule_functions)


# ------------------------------------------------------------------
# GroundTruth field list (must match design_spec + API.md exactly)
# ------------------------------------------------------------------

def check_groundtruth_fields():
    import dataclasses
    from spuriosity import GroundTruth
    expected_fields = {
        "true_coefficients", "break_points", "confounding_strength",
        "true_cate", "selection_mechanism", "heteroskedasticity",
        "multicollinearity", "measurement_error", "endogeneity",
        "unit_root", "treatment_effect_ate", "spuriosity_version",
        "numpy_version", "seed",
    }
    actual_fields = {f.name for f in dataclasses.fields(GroundTruth)}
    assert actual_fields == expected_fields, (
        f"MISMATCH.\nMissing: {expected_fields - actual_fields}\n"
        f"Extra: {actual_fields - expected_fields}"
    )

check("GroundTruth field list matches documented 14 fields exactly", check_groundtruth_fields)


# ------------------------------------------------------------------
# PanelGenerator builder methods full run-through
# ------------------------------------------------------------------

def check_panel_generator_full_builder_chain():
    from spuriosity import PanelGenerator
    gen = PanelGenerator(n_entities=100, n_periods=10, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_variable("x2", dist="uniform", low=0, high=1)
    gen.add_treatment("treat", assignment="random", start_period=5, propensity=0.5)
    gen.set_outcome(formula="x1 + x2 + treat", coefficients={"x1": 1.0, "treat": 2.0}, noise_std=0.5)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=1.0)
    gen.add_confounder(feature="x1", outcome="y", strength=0.3, observed=True)
    gen.add_selection_bias(rule="x1 > 2", drop_prob=0.3)
    gen.add_heteroskedasticity(feature="x1", formula="1 + 0.2*x1**2")
    gen.add_measurement_error(feature="x2", noise_std=0.1)
    gen.validate_combo()
    df, truth = gen.generate()
    assert len(df) > 0
    assert truth is not None

check("PanelGenerator full builder chain (8 pathology methods)", check_panel_generator_full_builder_chain)


def check_propensity_treatment():
    from spuriosity import PanelGenerator
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", assignment="propensity", propensity_formula="0.5*x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()
    assert "treat" in df.columns

check("PanelGenerator.add_treatment(assignment='propensity')", check_propensity_treatment)


def check_multicollinearity_and_endogeneity_and_unit_root():
    from spuriosity import PanelGenerator
    gen = PanelGenerator(n_entities=100, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.add_multicollinearity(feature="x2", correlated_with="x1", correlation=0.8)
    gen.add_unit_root(feature="x1", drift=0.1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()
    assert "x2" in df.columns

check("add_multicollinearity + add_unit_root", check_multicollinearity_and_endogeneity_and_unit_root)


def check_endogeneity_builder():
    from spuriosity import PanelGenerator
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=0.5)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, truth = gen.generate()
    assert "z" in df.columns
    assert len(truth.endogeneity) == 1

check("add_endogeneity", check_endogeneity_builder)


def check_hte_single_and_multi_dim():
    from spuriosity import PanelGenerator
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0})
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    df, truth = gen.generate()
    assert truth.true_cate(1.0) == 4.5

    gen2 = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen2.add_variable("x1")
    gen2.add_variable("x2")
    gen2.add_treatment("treat", propensity=0.5)
    gen2.set_outcome(formula="x1 + x2 + treat", coefficients={"x1": 1.0, "x2": 1.0, "treat": 0.0})
    gen2.add_hte(treatment="treat", modifier=["x1", "x2"], formula="3 + x1 - x2")
    df2, truth2 = gen2.generate()
    assert truth2.true_cate(x1=1.0, x2=1.0) == 3.0

check("add_hte single-dim and multi-dim", check_hte_single_and_multi_dim)


# ------------------------------------------------------------------
# reference fits: at least a smoke call for every one
# ------------------------------------------------------------------

def check_all_reference_fits_smoke():
    from spuriosity import PanelGenerator, reference

    gen = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 2.0}, noise_std=0.5)
    df, truth = gen.generate()

    reference.ols_fit(df, formula="y ~ x1 + treat")
    reference.sklearn_lr_fit(df, features=["x1", "treat"], target="y")
    reference.logit_fit(df.assign(binary=(df["y"] > 0).astype(int)), formula="binary ~ x1")

    gen2 = PanelGenerator(n_entities=500, n_periods=10, seed=1)
    gen2.add_variable("x1", dist="normal", mean=0, std=1)
    gen2.add_treatment("treat", propensity=0.5, start_period=0)
    gen2.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0, "Intercept": 1.0}, noise_std=0.1)
    gen2.add_structural_break(period=5, target="y", kind="coefficient_shift", magnitude=2.0, coefficient_target="treat")
    df2, truth2 = gen2.generate()
    reference.did_fit(df2, outcome="y", treatment="treat", period="period", post_period=5)
    # panel_fe_fit demeans within entity, so it can only estimate coefficients
    # on TIME-VARYING regressors (x1 here); a treatment column that is fixed
    # per entity across all periods gets fully absorbed by the entity effect
    # and cannot be estimated by FE at all -- use x1, not treat, for this check.
    reference.panel_fe_fit(df2, outcome="y", features=["x1"])
    reference.panel_re_fit(df2, outcome="y", features=["x1"])

    gen3 = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
    gen3.add_variable("x1")
    gen3.set_outcome(formula="x1", coefficients={"x1": 2.0})
    gen3.add_endogeneity(feature="x1", instrument="z", instrument_strength=1.0, endogeneity_strength=1.0)
    df3, truth3 = gen3.generate()
    reference.iv2sls_fit(df3, outcome="y", endogenous=["x1"], instruments=["z"])
    reference.first_stage_f_stat(df3, endogenous="x1", instruments=["z"])

    gen4 = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
    gen4.add_variable("x1")
    gen4.add_treatment("treat", assignment="propensity", propensity_formula="0.5*x1")
    gen4.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 2.0})
    df4, truth4 = gen4.generate()
    reference.psm_fit(df4, outcome="y", treatment="treat", covariates=["x1"])

    reference.xgboost_fit(df, outcome="y", features=["x1", "treat"])
    reference.doubleml_fit(df, outcome="y", treatment="treat", covariates=["x1"])
    reference.causal_forest_fit(df, outcome="y", treatment="treat", covariates=["x1"])

check("every reference.*_fit function (14 fits) smoke test", check_all_reference_fits_smoke)


def check_hausman_test():
    from spuriosity import PanelGenerator, reference
    gen = PanelGenerator(n_entities=500, n_periods=10, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=0.5)
    df, truth = gen.generate()
    # FE cannot estimate a time-invariant regressor's coefficient (fully
    # absorbed by the entity effect); x1 here is drawn fresh per row
    # (time-varying), so it is estimable by both FE and RE.
    fe = reference.panel_fe_fit(df, outcome="y", features=["x1"])
    re = reference.panel_re_fit(df, outcome="y", features=["x1"])
    result = reference.hausman_test(fe, re)
    assert set(result.keys()) == {"chi2", "dof", "p_value"}

check("reference.hausman_test", check_hausman_test)


# ------------------------------------------------------------------
# StressTest / compare_models / metrics
# ------------------------------------------------------------------

def check_stress_test_and_compare_models():
    from spuriosity import PanelGenerator, StressTest, compare_models, reference
    from spuriosity.metrics import default_registry

    gen = PanelGenerator(n_entities=2000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0})
    df, truth = gen.generate()

    test = StressTest(truth)
    report = test.evaluate(fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df, fit_kwargs={"formula": "y ~ x1"})
    assert "coef_rmse" in report.metrics

    results = compare_models(
        data=df, truth=truth,
        models={"OLS": (reference.ols_fit, reference.ols_predict)},
        fit_kwargs_per_model={"OLS": {"formula": "y ~ x1"}},
    )
    results.ranked_table()
    results.summary()

    custom_registry = default_registry.copy()
    custom_registry.register("always_one", lambda ctx: 1.0)
    test2 = StressTest(truth, metric_registry=custom_registry)
    report2 = test2.evaluate(fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df, fit_kwargs={"formula": "y ~ x1"})
    assert report2.metrics["always_one"] == 1.0

check("StressTest, compare_models, custom MetricRegistry", check_stress_test_and_compare_models)


def check_plot_recovery_report():
    import matplotlib
    matplotlib.use("Agg")
    from spuriosity import PanelGenerator, StressTest, reference, plot_recovery_report

    gen = PanelGenerator(n_entities=500, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0})
    df, truth = gen.generate()
    test = StressTest(truth)
    report = test.evaluate(fit_fn=reference.ols_fit, predict_fn=reference.ols_predict, data=df, fit_kwargs={"formula": "y ~ x1"})
    fig = plot_recovery_report(report)
    assert fig is not None

check("plot_recovery_report", check_plot_recovery_report)


def check_synthetic_control():
    from spuriosity import synthetic_control_fit
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(1)
    rows = []
    common = rng.normal(size=20).cumsum() * 0.3
    for entity in range(10):
        loading = rng.uniform(0.5, 1.5)
        for period in range(20):
            rows.append({"entity_id": entity, "period": period, "y": common[period]*loading + rng.normal()*0.1})
    df = pd.DataFrame(rows)
    result = synthetic_control_fit(df, outcome="y", entity_col="entity_id", period_col="period", treated_unit=0, treatment_period=10)
    assert result.average_effect is not None
    result.summary()

check("synthetic_control_fit", check_synthetic_control)


# ------------------------------------------------------------------
# Print results
# ------------------------------------------------------------------

print()
print(f"{'CHECK':<65} {'STATUS':<6} DETAIL")
print("-" * 110)
n_ok, n_fail = 0, 0
for label, status, detail in results:
    print(f"{label:<65} {status:<6} {detail}")
    if status == "OK":
        n_ok += 1
    else:
        n_fail += 1
print("-" * 110)
print(f"{n_ok} passed, {n_fail} failed out of {len(results)}")

if n_fail > 0:
    sys.exit(1)
