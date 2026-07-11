"""Tests for structural break integration in spuriosity.generator.PanelGenerator.

These use large N and tight tolerances to verify the injected pathology
actually has the claimed statistical effect, not just that generation runs
without error.
"""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity import PanelGenerator


def test_mean_shift_changes_outcome_mean_by_magnitude():
    gen = PanelGenerator(n_entities=1000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 2.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=10.0)
    df, truth = gen.generate()

    pre_mean = df[df["period"] < 5]["y"].mean()
    post_mean = df[df["period"] >= 5]["y"].mean()
    assert (post_mean - pre_mean) == pytest.approx(10.0, abs=0.5)


def test_mean_shift_recorded_in_ground_truth():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=3.0)
    _, truth = gen.generate()
    assert len(truth.break_points) == 1
    assert truth.break_points[0].period == 5
    assert truth.break_points[0].magnitude == 3.0


def test_variance_shift_changes_residual_std_by_factor():
    gen = PanelGenerator(n_entities=2000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
    gen.add_structural_break(period=5, target="y", kind="variance_shift", magnitude=3.0)
    df, _ = gen.generate()

    resid_pre = df[df["period"] < 5]["y"] - df[df["period"] < 5]["x1"]
    resid_post = df[df["period"] >= 5]["y"] - df[df["period"] >= 5]["x1"]
    ratio = resid_post.std() / resid_pre.std()
    assert ratio == pytest.approx(3.0, abs=0.3)


def test_coefficient_shift_changes_slope():
    gen = PanelGenerator(n_entities=2000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=0.1)
    gen.add_structural_break(
        period=5, target="y", kind="coefficient_shift", magnitude=5.0, coefficient_target="x1"
    )
    df, _ = gen.generate()

    pre = df[df["period"] < 5]
    post = df[df["period"] >= 5]
    slope_pre = np.polyfit(pre["x1"], pre["y"], 1)[0]
    slope_post = np.polyfit(post["x1"], post["y"], 1)[0]
    assert slope_pre == pytest.approx(1.0, abs=0.2)
    assert slope_post == pytest.approx(5.0, abs=0.2)


def test_coefficient_shift_with_fn_outcome_raises_clear_error():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(fn=lambda x1: x1, noise_std=0.0)
    gen.add_structural_break(
        period=3, target="y", kind="coefficient_shift", magnitude=5.0, coefficient_target="x1"
    )
    with pytest.raises(ValueError, match="coefficient_shift requires a formula-specified outcome"):
        gen.generate()


def test_coefficient_shift_unknown_target_raises_clear_error():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_structural_break(
        period=3, target="y", kind="coefficient_shift", magnitude=5.0, coefficient_target="nonexistent"
    )
    with pytest.raises(ValueError, match="is not a column of the outcome design matrix"):
        gen.generate()


def test_add_structural_break_rejects_out_of_range_period():
    gen = PanelGenerator(n_entities=10, n_periods=5, seed=1)
    with pytest.raises(ValueError):
        gen.add_structural_break(period=100, target="y", kind="mean_shift", magnitude=1.0)


def test_validate_combo_integration_prints_and_returns_warnings(capsys):
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_structural_break(period=3, target="y", kind="mean_shift", magnitude=1.0)
    gen.add_structural_break(period=3, target="y", kind="mean_shift", magnitude=2.0)

    warnings = gen.validate_combo()
    assert len(warnings) == 1

    captured = capsys.readouterr()
    assert "spuriosity warning" in captured.out


def test_validate_combo_no_pathologies_no_warnings():
    gen = PanelGenerator(n_entities=10, n_periods=10, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    assert gen.validate_combo() == []


def test_structural_break_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=100, n_periods=10, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=2.0)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    import pandas.testing as pdt

    pdt.assert_frame_equal(df_a, df_b)


def test_multiple_non_conflicting_breaks_both_apply():
    gen = PanelGenerator(n_entities=1000, n_periods=15, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=5.0)
    gen.add_structural_break(period=10, target="y", kind="mean_shift", magnitude=5.0)
    df, truth = gen.generate()

    seg1 = df[df["period"] < 5]["y"].mean()
    seg2 = df[(df["period"] >= 5) & (df["period"] < 10)]["y"].mean()
    seg3 = df[df["period"] >= 10]["y"].mean()

    assert (seg2 - seg1) == pytest.approx(5.0, abs=0.5)
    assert (seg3 - seg2) == pytest.approx(5.0, abs=0.5)
    assert len(truth.break_points) == 2
