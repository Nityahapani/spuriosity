"""Tests for multi-dimensional HTE integration in
spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator


def _binned_treatment_effect(
    df: pd.DataFrame, x1_center: float, x2_center: float, half_width: float = 0.15
) -> float:
    window = df[
        (df["x1"] >= x1_center - half_width)
        & (df["x1"] <= x1_center + half_width)
        & (df["x2"] >= x2_center - half_width)
        & (df["x2"] <= x2_center + half_width)
    ]
    treated = window[window["treat"] == 1]["y"].mean()
    control = window[window["treat"] == 0]["y"].mean()
    return float(treated - control)


def _build_multidim_hte_generator(n_entities: int = 1_000_000, seed: int = 42) -> PanelGenerator:
    gen = PanelGenerator(n_entities=n_entities, n_periods=1, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_variable("x2", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + x2 + treat",
        coefficients={"x1": 1.0, "x2": 1.0, "treat": 0.0, "Intercept": 0.0},
        noise_std=0.1,
    )
    gen.add_hte(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    return gen


def test_multidim_hte_effect_varies_with_both_modifiers():
    df, _ = _build_multidim_hte_generator().generate()
    effect_at_origin = _binned_treatment_effect(df, 0.0, 0.0)
    effect_elsewhere = _binned_treatment_effect(df, 2.0, 1.0)
    assert effect_at_origin == pytest.approx(3.0, abs=0.5)
    assert effect_elsewhere == pytest.approx(5.5, abs=0.7)


def test_multidim_ground_truth_true_cate_matches_formula():
    _, truth = _build_multidim_hte_generator(n_entities=100).generate()
    assert truth.true_cate is not None
    assert truth.true_cate(x1=0.0, x2=0.0) == pytest.approx(3.0)
    assert truth.true_cate(x1=2.0, x2=1.0) == pytest.approx(5.5)


def test_multidim_ground_truth_treatment_effect_ate_is_average_cate():
    _, truth = _build_multidim_hte_generator(n_entities=500_000).generate()
    # E[x1] = E[x2] = 0 for standard normal -> E[3 + 1.5*x1 - 0.5*x2] = 3
    assert truth.treatment_effect_ate == pytest.approx(3.0, abs=0.05)


def test_multidim_hte_reproducible_with_same_seed():
    df_a, _ = _build_multidim_hte_generator(n_entities=1000, seed=7).generate()
    df_b, _ = _build_multidim_hte_generator(n_entities=1000, seed=7).generate()
    pdt.assert_frame_equal(df_a, df_b)


def test_multidim_hte_missing_one_of_two_modifiers_raises():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")  # x2 not declared
    gen.add_treatment("treat", propensity=0.5)
    gen.set_outcome(formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0})
    gen.add_hte(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    with pytest.raises(ValueError, match=r"\['x2'\]"):
        gen.generate()


def test_multidim_hte_composes_with_confounder_and_structural_break():
    gen = PanelGenerator(n_entities=200_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_variable("x2", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + x2 + treat",
        coefficients={"x1": 1.0, "x2": 1.0, "treat": 0.0, "Intercept": 0.0},
        noise_std=0.1,
    )
    gen.add_hte(treatment="treat", modifier=["x1", "x2"], formula="3 + 1.5*x1 - 0.5*x2")
    gen.add_confounder(feature="x1", outcome="y", strength=0.2, observed=True)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=4.0)

    df, truth = gen.generate()
    assert truth.true_cate is not None
    assert "_confounder_x1" in df.columns
    assert len(truth.break_points) == 1

    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(4.0, abs=0.5)


def test_backward_compat_single_string_modifier_still_works_end_to_end():
    """Explicit end-to-end regression check: v1's single-string modifier
    call pattern must produce identical results after the multi-dim
    extension as it did before."""
    gen = PanelGenerator(n_entities=500_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", propensity=0.5, start_period=0)
    gen.set_outcome(
        formula="x1 + treat", coefficients={"x1": 1.0, "treat": 0.0, "Intercept": 0.0}, noise_std=0.1
    )
    gen.add_hte(treatment="treat", modifier="x1", formula="3 + 1.5*x1")
    df, truth = gen.generate()

    assert truth.true_cate(0.0) == pytest.approx(3.0)
    assert truth.true_cate(2.0) == pytest.approx(6.0)
    assert truth.treatment_effect_ate == pytest.approx(3.0, abs=0.05)
