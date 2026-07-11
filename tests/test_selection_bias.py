"""Tests for spuriosity.pathologies.SelectionBias (isolated, non-generator logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spuriosity.ground_truth import SelectionInfo
from spuriosity.pathologies import SelectionBias


def test_construction_rejects_invalid_drop_prob():
    with pytest.raises(ValueError):
        SelectionBias(rule="x1 > 1", drop_prob=1.5)
    with pytest.raises(ValueError):
        SelectionBias(rule="x1 > 1", drop_prob=-0.1)


def test_compute_mask_to_drop_only_flags_matching_rows_with_drop_prob_one():
    sel = SelectionBias(rule="x1 > 1.5", drop_prob=1.0)
    df = pd.DataFrame({"x1": [0.0, 2.0, 3.0, -1.0]})
    rng = np.random.default_rng(0)
    mask = sel.compute_mask_to_drop(df, rng)
    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_compute_mask_to_drop_prob_zero_drops_nothing():
    sel = SelectionBias(rule="x1 > 1.5", drop_prob=0.0)
    df = pd.DataFrame({"x1": [0.0, 2.0, 3.0, -1.0]})
    rng = np.random.default_rng(0)
    mask = sel.compute_mask_to_drop(df, rng)
    assert not mask.any()


def test_compute_mask_to_drop_prob_partial_is_probabilistic():
    sel = SelectionBias(rule="x1 > 0", drop_prob=0.5)
    n = 200_000
    df = pd.DataFrame({"x1": np.ones(n)})  # all rows match the rule
    rng = np.random.default_rng(0)
    mask = sel.compute_mask_to_drop(df, rng)
    assert mask.mean() == pytest.approx(0.5, abs=0.01)


def test_non_boolean_rule_raises():
    sel = SelectionBias(rule="x1 + 1", drop_prob=0.5)
    df = pd.DataFrame({"x1": [1.0, 2.0]})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="did not evaluate to a boolean result"):
        sel.compute_mask_to_drop(df, rng)


def test_unknown_column_in_rule_raises():
    sel = SelectionBias(rule="nonexistent > 1", drop_prob=0.5)
    df = pd.DataFrame({"x1": [1.0, 2.0]})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Failed to evaluate selection rule"):
        sel.compute_mask_to_drop(df, rng)


def test_code_injection_attempt_is_blocked():
    sel = SelectionBias(rule='__import__("os").system("echo pwned")', drop_prob=0.5)
    df = pd.DataFrame({"x1": [1.0, 2.0]})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Failed to evaluate selection rule"):
        sel.compute_mask_to_drop(df, rng)


def test_rule_can_reference_multiple_columns():
    sel = SelectionBias(rule="x1 > 0 and x2 < 0", drop_prob=1.0)
    df = pd.DataFrame({"x1": [1.0, 1.0, -1.0], "x2": [-1.0, 1.0, -1.0]})
    rng = np.random.default_rng(0)
    mask = sel.compute_mask_to_drop(df, rng)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_ground_truth_contribution():
    sel = SelectionBias(rule="x1 > 1.5", drop_prob=0.4)
    contrib = sel.ground_truth_contribution()
    assert contrib == {"selection_mechanism": SelectionInfo(rule="x1 > 1.5", drop_prob=0.4)}
