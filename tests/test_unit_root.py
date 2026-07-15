"""Tests for spuriosity.pathologies.UnitRoot (isolated, non-generator logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spuriosity.ground_truth import UnitRootInfo
from spuriosity.pathologies import UnitRoot


def test_apply_to_panel_single_entity_cumsum():
    df = pd.DataFrame({
        "entity_id": [0, 0, 0],
        "period": [0, 1, 2],
        "x1": [1.0, 1.0, 1.0],
    })
    ur = UnitRoot(feature="x1")
    result = ur.apply_to_panel(df)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_apply_to_panel_resets_at_entity_boundary():
    df = pd.DataFrame({
        "entity_id": [0, 0, 0, 1, 1, 1],
        "period": [0, 1, 2, 0, 1, 2],
        "x1": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    })
    ur = UnitRoot(feature="x1")
    result = ur.apply_to_panel(df)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 2.0, 4.0, 6.0])


def test_apply_to_panel_with_drift():
    df = pd.DataFrame({
        "entity_id": [0, 0, 0],
        "period": [0, 1, 2],
        "x1": [1.0, 1.0, 1.0],
    })
    ur = UnitRoot(feature="x1", drift=0.5)
    result = ur.apply_to_panel(df)
    np.testing.assert_allclose(result, [1.5, 3.0, 4.5])


def test_apply_to_panel_zero_drift_is_default():
    ur = UnitRoot(feature="x1")
    assert ur.drift == 0.0


def test_apply_to_panel_preserves_row_order():
    """Row order must be preserved even if entities appear interleaved or
    out of period order in the input DataFrame (defensive check -- normal
    PanelGenerator output is always sorted, but the method itself
    shouldn't silently reorder)."""
    df = pd.DataFrame({
        "entity_id": [0, 1, 0, 1],
        "period": [0, 0, 1, 1],
        "x1": [1.0, 10.0, 1.0, 10.0],
    })
    ur = UnitRoot(feature="x1")
    result = ur.apply_to_panel(df)
    # entity 0: rows 0, 2 -> cumsum [1, 2]; entity 1: rows 1, 3 -> cumsum [10, 20]
    np.testing.assert_allclose(result, [1.0, 10.0, 2.0, 20.0])


def test_apply_to_panel_missing_feature_raises():
    df = pd.DataFrame({"entity_id": [0], "period": [0]})
    ur = UnitRoot(feature="nonexistent")
    with pytest.raises(ValueError, match="not present in the data"):
        ur.apply_to_panel(df)


def test_ground_truth_contribution():
    ur = UnitRoot(feature="x1", drift=0.2)
    contrib = ur.ground_truth_contribution()
    assert contrib == {"unit_root": [UnitRootInfo(feature="x1", drift=0.2)]}


def test_variance_grows_with_time_index():
    """A defining property of a random walk: Var(x_t) grows linearly with
    t (= t * increment_variance for a driftless walk starting at 0),
    unlike an i.i.d. series where variance is constant."""
    rng = np.random.default_rng(0)
    n_entities = 5000
    n_periods = 50
    entity_id = np.repeat(np.arange(n_entities), n_periods)
    period = np.tile(np.arange(n_periods), n_entities)
    x1 = rng.normal(size=n_entities * n_periods)
    df = pd.DataFrame({"entity_id": entity_id, "period": period, "x1": x1})

    ur = UnitRoot(feature="x1")
    walk = ur.apply_to_panel(df)
    df["walk"] = walk

    var_early = df[df["period"] == 5]["walk"].var()
    var_late = df[df["period"] == 45]["walk"].var()
    assert var_late > var_early * 5  # should grow roughly linearly with t


def test_adf_test_fails_to_reject_unit_root():
    """Formal diagnostic check: the Augmented Dickey-Fuller test should
    fail to reject the unit-root null for a generated random walk, while
    correctly rejecting it for the underlying i.i.d. increments."""
    from statsmodels.tsa.stattools import adfuller

    rng = np.random.default_rng(0)
    n = 500
    entity_id = np.zeros(n, dtype=int)
    period = np.arange(n)
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"entity_id": entity_id, "period": period, "x1": x1})

    ur = UnitRoot(feature="x1")
    walk = ur.apply_to_panel(df)

    adf_walk = adfuller(walk)
    adf_iid = adfuller(x1)

    assert adf_walk[1] > 0.1  # fail to reject unit root
    assert adf_iid[1] < 0.05  # correctly reject unit root for i.i.d. data
