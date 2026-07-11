"""Tests for SelectionBias integration in spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import warnings

import pandas.testing as pdt
import pytest
from scipy import stats

from spuriosity import PanelGenerator


def test_selection_bias_reduces_row_count():
    gen = PanelGenerator(n_entities=10_000, n_periods=1, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="x1 > 1.5", drop_prob=1.0)
    df, _ = gen.generate()
    assert len(df) < 10_000
    assert (df["x1"] > 1.5).sum() == 0


def test_selection_bias_matches_expected_removal_fraction():
    n = 100_000
    gen = PanelGenerator(n_entities=n, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=0.1)
    gen.add_selection_bias(rule="x1 > 1.5", drop_prob=0.4)
    df, _ = gen.generate()

    base_rate = 1 - stats.norm.cdf(1.5)  # P(x1 > 1.5) for standard normal
    expected_remaining_fraction = base_rate * (1 - 0.4)
    observed_remaining_fraction = (df["x1"] > 1.5).mean()
    assert observed_remaining_fraction == pytest.approx(expected_remaining_fraction, abs=0.005)


def test_drop_prob_zero_removes_nothing():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="x1 > 1.5", drop_prob=0.0)
    df, _ = gen.generate()
    assert len(df) == 1000


def test_outcome_dependent_selection_survivorship_bias():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=0.1)
    gen.add_selection_bias(rule="y < 0", drop_prob=0.7)
    df, _ = gen.generate()
    # Selection on the outcome itself should shift the remaining sample's mean upward
    assert df["y"].mean() > 0.1


def test_non_boolean_rule_raises_at_generate_time():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="x1 + 1", drop_prob=0.5)
    with pytest.raises(ValueError, match="did not evaluate to a boolean result"):
        gen.generate()


def test_unknown_column_raises_at_generate_time():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="nonexistent > 1", drop_prob=0.5)
    with pytest.raises(ValueError, match="Failed to evaluate selection rule"):
        gen.generate()


def test_ground_truth_records_selection_mechanism():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="x1 > 1.5", drop_prob=0.4)
    _, truth = gen.generate()
    assert truth.selection_mechanism is not None
    assert truth.selection_mechanism.rule == "x1 > 1.5"
    assert truth.selection_mechanism.drop_prob == 0.4


def test_ground_truth_selection_mechanism_none_without_selection_bias():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.selection_mechanism is None


def test_multiple_selection_biases_warns_and_records_only_first():
    gen = PanelGenerator(n_entities=100_000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=0.1)
    gen.add_selection_bias(rule="x1 > 1.5", drop_prob=0.5)
    gen.add_selection_bias(rule="x1 < -1.5", drop_prob=0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df, truth = gen.generate()
        assert len(caught) == 1
        assert "2 selection_bias pathologies were added" in str(caught[0].message)

    # both rules should still have been applied to the actual data
    assert (df["x1"] > 1.5).sum() < (df["x1"] < -1.5).sum() + 100  # sanity, both trimmed
    assert truth.selection_mechanism.rule == "x1 > 1.5"  # only first recorded


def test_single_selection_bias_does_not_warn():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="x1 > 1.5", drop_prob=0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.generate()
        assert len(caught) == 0


def test_selection_bias_reproducible_with_same_seed():
    def build():
        g = PanelGenerator(n_entities=1000, n_periods=1, seed=7)
        g.add_variable("x1")
        g.set_outcome(formula="x1", coefficients={"x1": 1.0}, noise_std=1.0)
        g.add_selection_bias(rule="x1 > 0.5", drop_prob=0.5)
        return g.generate()

    df_a, _ = build()
    df_b, _ = build()
    pdt.assert_frame_equal(df_a, df_b)


def test_selection_bias_composes_with_structural_break_and_confounder():
    gen = PanelGenerator(n_entities=100_000, n_periods=10, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0, "Intercept": 0.0}, noise_std=0.1)
    gen.add_confounder(feature="x1", outcome="y", strength=0.3, observed=True)
    gen.add_structural_break(period=5, target="y", kind="mean_shift", magnitude=3.0)
    gen.add_selection_bias(rule="x1 > 2.0", drop_prob=0.5)

    df, truth = gen.generate()
    assert "_confounder_x1" in df.columns
    assert len(truth.break_points) == 1
    assert truth.selection_mechanism is not None
    assert len(df) < 1_000_000  # some rows removed by selection


def test_index_is_reset_after_row_removal():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_selection_bias(rule="x1 > 0", drop_prob=1.0)
    df, _ = gen.generate()
    assert list(df.index) == list(range(len(df)))
