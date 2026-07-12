"""Tests for HTE integration in spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import warnings

import pandas as pd
import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator


def _binned_treatment_effect(df: pd.DataFrame, x1_center: float, half_width: float = 0.1) -> float:
    """Crude binned CATE estimator: mean(y | treat=1) - mean(y | treat=0)
    within a narrow window around x1_center. Used to verify the true
    treatment effect actually varies in the generated data as specified."""
    window = df[(df["x1"] >= x1_center - half_width) & (df["x1"] <= x1_center + half_width)]
    treated = window[window["treat"] == 1]["y"].mean()
    control = window[window["treat"] == 0]["y"].mean()
    return float(treated - control)


def _build_hte_generator(n_entities: int = 500_000, seed: int = 42) -> PanelGenerator:
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + treat",
        coefficients={"x1": 1.0, "treat": 0.0, "Intercept": 0.0},
        noise_std=0.1,
    )
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    return gen


def test_hte_effect_varies_with_modifier():
    df, _ = _build_hte_generator().generate()
    effect_at_0 = _binned_treatment_effect(df, x1_center=0.0)
    effect_at_2 = _binned_treatment_effect(df, x1_center=2.0)
    assert effect_at_0 == pytest.approx(3.0, abs=0.3)
    assert effect_at_2 == pytest.approx(6.0, abs=0.5)


def test_ground_truth_true_cate_matches_formula():
    _, truth = _build_hte_generator(n_entities=100).generate()
    assert truth.true_cate is not None
    assert truth.true_cate(0.0) == pytest.approx(3.0)
    assert truth.true_cate(2.0) == pytest.approx(6.0)


def test_ground_truth_treatment_effect_ate_is_average_cate():
    _, truth = _build_hte_generator(n_entities=500_000).generate()
    # E[x1] = 0 for standard normal, so E[3 + 1.5*x1] = 3
    assert truth.treatment_effect_ate == pytest.approx(3.0, abs=0.05)


def test_hte_requires_formula_outcome_not_fn():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(fn=lambda x1, treat: x1 + treat, noise_std=0.1)
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    with pytest.raises(ValueError, match="requires an outcome specified via set_outcome"):
        gen.generate()


def test_hte_requires_declared_treatment():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_hte(treatment="nonexistent_treat", modifier="x1", formula="3 + 1.5*x1")
    with pytest.raises(ValueError, match="no treatment with that name was declared"):
        gen.generate()


def test_hte_requires_declared_modifier():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="treat", coefficients={"treat": 0.0})
    gen.add_hte(treatment="treat", modifier="nonexistent_x", formula="3 + 1.5*nonexistent_x")
    with pytest.raises(ValueError, match="is not a declared variable, treatment, or reserved column"):
        gen.generate()


def test_hte_requires_treatment_as_explicit_formula_term():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})  # treat omitted from formula
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    with pytest.raises(ValueError, match="does not appear as a term in the outcome formula"):
        gen.generate()


def test_hte_warns_when_fixed_treatment_coefficient_also_supplied():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 5.0})
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.generate()
        assert len(caught) == 1
        assert "replaces the effect of treatment" in str(caught[0].message)


def test_hte_no_warning_when_treatment_coefficient_is_zero():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0})
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.generate()
        assert len(caught) == 0


def test_hte_reproducible_with_same_seed():
    df_a, _ = _build_hte_generator(n_entities=1000, seed=7).generate()
    df_b, _ = _build_hte_generator(n_entities=1000, seed=7).generate()
    pdt.assert_frame_equal(df_a, df_b)


def test_ground_truth_true_cate_none_without_hte():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.true_cate is None


def test_hte_composes_with_confounder_and_structural_break():
    gen = PanelGenerator(n_entities=200_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0, "Intercept": 0.0}, noise_std=0.1
    )
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    gen.add_confounder(feature="x1", outcome="y", strength=0.2, observed=True)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=4.0)

    df, truth = gen.generate()
    assert truth.true_cate is not None
    assert "_confounder_x1" in df.columns
    assert len(truth.break_points) == 1

    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(4.0, abs=0.5)
