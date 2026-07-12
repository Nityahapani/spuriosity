"""Tests for spuriosity.ground_truth."""

from __future__ import annotations

import json

import pytest

from spuriosity.ground_truth import BreakInfo, GroundTruth, SelectionInfo


def test_minimal_ground_truth_defaults():
    gt = GroundTruth(true_coefficients={"x1": 2.0})
    assert gt.break_points == []
    assert gt.confounding_strength == {}
    assert gt.true_cate is None
    assert gt.selection_mechanism is None
    assert gt.treatment_effect_ate is None
    assert gt.seed == 0


def test_to_dict_minimal():
    gt = GroundTruth(true_coefficients={"x1": 2.0, "x2": 0.5})
    d = gt.to_dict()
    assert d["true_coefficients"] == {"x1": 2.0, "x2": 0.5}
    assert d["break_points"] == []
    assert d["has_true_cate"] is False
    assert d["selection_mechanism"] is None


def test_to_dict_full():
    gt = GroundTruth(
        true_coefficients={"x1": 2.0, "treat": 3.0},
        break_points=[BreakInfo(period=20, target="y", kind="mean_shift", magnitude=2.0)],
        confounding_strength={"x1": 0.6},
        true_cate=lambda x1: 3 + 1.5 * x1,
        selection_mechanism=SelectionInfo(rule="x1 > 1.5", drop_prob=0.4),
        treatment_effect_ate=3.0,
        spuriosity_version="0.1.0",
        numpy_version="1.26.0",
        seed=42,
    )
    d = gt.to_dict()
    assert d["has_true_cate"] is True
    assert d["break_points"][0] == {
        "period": 20,
        "target": "y",
        "kind": "mean_shift",
        "magnitude": 2.0,
    }


# ----------------------------------------------------------------------
# __repr__
# ----------------------------------------------------------------------


def test_repr_includes_seed_and_version():
    gt = GroundTruth(true_coefficients={"x1": 2.0}, seed=42, spuriosity_version="0.1.0")
    r = repr(gt)
    assert "GroundTruth" in r
    assert "seed=42" in r
    assert "'0.1.0'" in r  # spuriosity_version as repr-style quoted string


def test_repr_lists_all_coefficients():
    gt = GroundTruth(true_coefficients={"x1": 2.0, "x2": 0.5, "treat": 3.0})
    r = repr(gt)
    # Each coefficient should appear as `key: value`
    assert "'x1': 2.0" in r or '"x1": 2.0' in r
    assert "2.0" in r
    assert "0.5" in r
    assert "3.0" in r


def test_repr_handles_minimal_ground_truth_gracefully():
    gt = GroundTruth(true_coefficients={})
    r = repr(gt)
    assert "GroundTruth" in r
    assert "true_coefficients: <empty>" in r


def test_repr_marks_optional_fields_only_when_present():
    gt_no_optionals = GroundTruth(true_coefficients={"x1": 1.0})
    r = repr(gt_no_optionals)
    # No break_points, no confounding, no selection, no ATE, no CATE
    assert "break_points" not in r
    assert "confounding_strength" not in r
    assert "selection_mechanism" not in r
    assert "treatment_effect_ate" not in r
    # but has_true_cate: False should appear
    assert "has_true_cate: False" in r

    gt_with_break = GroundTruth(
        true_coefficients={"x1": 1.0},
        break_points=[BreakInfo(period=10, target="y", kind="mean_shift", magnitude=1.0)],
    )
    r2 = repr(gt_with_break)
    assert "break_points: 1" in r2
    assert "mean_shift" in r2


def test_repr_shows_cate_flag_when_true_cate_set():
    gt = GroundTruth(
        true_coefficients={"x1": 1.0, "treat": 0.0},
        true_cate=lambda x1: 1.0 + 0.5 * x1,
    )
    r = repr(gt)
    assert "has_true_cate: True" in r


def test_break_info_repr_leads_with_kind():
    """The new BreakInfo repr puts `kind` first so it's scannable in logs."""
    bi = BreakInfo(period=20, target="y", kind="coefficient_shift", magnitude=5.0)
    r = repr(bi)
    assert r.startswith("BreakInfo(kind=")
    assert "coefficient_shift" in r
    assert "period=20" in r


def test_selection_info_repr_includes_rule_and_prob():
    si = SelectionInfo(rule="y > 0", drop_prob=0.3)
    r = repr(si)
    assert r.startswith("SelectionInfo(")
    assert "'y > 0'" in r
    assert "drop_prob=0.3" in r


def test_true_cate_is_callable_and_not_serialized():
    gt = GroundTruth(
        true_coefficients={},
        true_cate=lambda x1: 3 + 1.5 * x1,
    )
    assert callable(gt.true_cate)
    assert gt.true_cate(1.0) == pytest.approx(4.5)
    assert "true_cate" not in gt.to_dict()
    assert "has_true_cate" in gt.to_dict()


def test_to_json_round_trips_through_json_loads():
    gt = GroundTruth(
        true_coefficients={"x1": 2.0},
        break_points=[BreakInfo(period=10, target="y", kind="mean_shift", magnitude=1.0)],
        seed=7,
    )
    parsed = json.loads(gt.to_json())
    assert parsed["seed"] == 7
    assert parsed["break_points"][0]["period"] == 10


def test_to_json_indent_none_is_compact():
    gt = GroundTruth(true_coefficients={"x1": 1.0})
    compact = gt.to_json(indent=None)
    assert "\n" not in compact
    # still valid JSON
    assert json.loads(compact)["true_coefficients"] == {"x1": 1.0}


def test_ground_truth_is_frozen():
    gt = GroundTruth(true_coefficients={"x1": 1.0})
    with pytest.raises(Exception):
        gt.seed = 99  # type: ignore[misc]


def test_break_info_is_frozen():
    b = BreakInfo(period=1, target="y", kind="mean_shift", magnitude=1.0)
    with pytest.raises(Exception):
        b.period = 2  # type: ignore[misc]


def test_selection_info_is_frozen():
    s = SelectionInfo(rule="x1 > 0", drop_prob=0.1)
    with pytest.raises(Exception):
        s.drop_prob = 0.5  # type: ignore[misc]
