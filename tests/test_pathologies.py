"""Tests for spuriosity.pathologies (StructuralBreak, validate_combo)."""

from __future__ import annotations

import pytest

from spuriosity.pathologies import StructuralBreak, validate_combo


def test_structural_break_rejects_unsupported_kind():
    with pytest.raises(ValueError):
        StructuralBreak(period=1, target="y", kind="banana", magnitude=1.0)  # type: ignore[arg-type]


def test_structural_break_rejects_negative_period():
    with pytest.raises(ValueError):
        StructuralBreak(period=-1, target="y", kind="mean_shift", magnitude=1.0)


def test_coefficient_shift_requires_coefficient_target():
    with pytest.raises(ValueError):
        StructuralBreak(period=1, target="y", kind="coefficient_shift", magnitude=1.0)


def test_ground_truth_contribution_shape():
    b = StructuralBreak(period=5, target="y", kind="mean_shift", magnitude=2.0)
    contrib = b.ground_truth_contribution()
    assert "break_points" in contrib
    assert len(contrib["break_points"]) == 1
    bp = contrib["break_points"][0]
    assert bp.period == 5
    assert bp.target == "y"
    assert bp.kind == "mean_shift"
    assert bp.magnitude == 2.0


def test_validate_combo_no_warnings_for_unrelated_breaks():
    b1 = StructuralBreak(period=1, target="y", kind="mean_shift", magnitude=1.0)
    b2 = StructuralBreak(period=5, target="y", kind="mean_shift", magnitude=1.0)
    assert validate_combo([b1, b2]) == []


def test_validate_combo_warns_on_same_period_same_target():
    b1 = StructuralBreak(period=5, target="y", kind="mean_shift", magnitude=1.0)
    b2 = StructuralBreak(period=5, target="y", kind="variance_shift", magnitude=2.0)
    warnings = validate_combo([b1, b2])
    assert len(warnings) == 1
    assert "same period" in warnings[0]


def test_validate_combo_no_warning_for_different_targets_same_period():
    b1 = StructuralBreak(period=5, target="y", kind="mean_shift", magnitude=1.0)
    b2 = StructuralBreak(period=5, target="z", kind="mean_shift", magnitude=1.0)
    assert validate_combo([b1, b2]) == []


def test_validate_combo_never_raises_only_returns_warnings():
    """v1 policy: permissive, warnings only, no hard errors from validate_combo."""
    breaks = [
        StructuralBreak(period=5, target="y", kind="mean_shift", magnitude=1.0)
        for _ in range(5)
    ]
    # Should not raise despite many conflicting breaks
    warnings = validate_combo(breaks)
    assert len(warnings) == 10  # C(5,2) pairs, all conflicting
