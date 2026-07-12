"""Tests for spuriosity.generator.PanelGenerator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from spuriosity import PanelGenerator


def _basic_generator(seed: int = 42) -> PanelGenerator:
    gen = PanelGenerator(n_entities=50, n_periods=10, seed=seed)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.add_variable("x2", dist="normal", mean=0, std=1)
    gen.add_treatment("treat", assignment="random", start_period=5, propensity=0.5)
    gen.set_outcome(
        formula="x1 + x2 + treat",
        coefficients={"x1": 2.0, "x2": 0.5, "treat": 3.0, "Intercept": 1.0},
        noise_std=1.0,
    )
    return gen


# ----------------------------------------------------------------------
# Shape and basic correctness
# ----------------------------------------------------------------------


def test_generate_shape_and_columns():
    gen = _basic_generator()
    df, truth = gen.generate()
    assert df.shape == (500, 6)
    assert list(df.columns) == ["entity_id", "period", "x1", "x2", "treat", "y"]


def test_generate_returns_dataframe_and_groundtruth_types():
    gen = _basic_generator()
    df, truth = gen.generate()
    assert isinstance(df, pd.DataFrame)
    from spuriosity import GroundTruth

    assert isinstance(truth, GroundTruth)


def test_entity_and_period_indexing_correct():
    gen = PanelGenerator(n_entities=3, n_periods=4, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, _ = gen.generate()
    assert sorted(df["entity_id"].unique()) == [0, 1, 2]
    assert sorted(df["period"].unique()) == [0, 1, 2, 3]
    assert len(df) == 12
    # Each entity should have exactly n_periods rows
    assert (df.groupby("entity_id").size() == 4).all()


def test_n_entities_one_is_pure_time_series():
    gen = PanelGenerator(n_entities=1, n_periods=20, seed=2)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df, _ = gen.generate()
    assert df["entity_id"].nunique() == 1
    assert len(df) == 20


# ----------------------------------------------------------------------
# Outcome specification: formula vs. fn
# ----------------------------------------------------------------------


def test_formula_outcome_uses_supplied_coefficients():
    gen = PanelGenerator(n_entities=2000, n_periods=1, seed=42)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(formula="x1", coefficients={"x1": 3.0, "Intercept": 5.0}, noise_std=0.01)
    df, truth = gen.generate()
    # With tiny noise and large N, recovered slope/intercept should be close to truth
    slope, intercept = np.polyfit(df["x1"], df["y"], 1)
    assert slope == pytest.approx(3.0, abs=0.05)
    assert intercept == pytest.approx(5.0, abs=0.05)
    assert truth.true_coefficients == {"x1": 3.0, "Intercept": 5.0}


def test_formula_missing_coefficient_defaults_to_zero():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_variable("x2")
    # x2's coefficient is not supplied -> should default to 0.0, not error
    gen.set_outcome(formula="x1 + x2", coefficients={"x1": 1.0}, noise_std=0.0)
    df, _ = gen.generate()
    # y should equal x1 + Intercept(0) exactly, since noise_std=0 and x2 coef=0
    np.testing.assert_allclose(df["y"].to_numpy(), df["x1"].to_numpy())


def test_fn_outcome_path():
    gen = PanelGenerator(n_entities=100, n_periods=1, seed=1)
    gen.add_variable("x1", dist="normal", mean=0, std=1)
    gen.set_outcome(fn=lambda x1: x1**2, noise_std=0.0)
    df, _ = gen.generate()
    np.testing.assert_allclose(df["y"].to_numpy(), df["x1"].to_numpy() ** 2)


def test_fn_outcome_receives_treatment_kwarg():
    gen = PanelGenerator(n_entities=100, n_periods=2, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", start_period=1, propensity=1.0)  # everyone treated from period 1

    received = {}

    def fn(x1, treat):
        received["treat"] = treat
        return x1 + treat

    gen.set_outcome(fn=fn, noise_std=0.0)
    df, _ = gen.generate()
    assert "treat" in received
    np.testing.assert_allclose(df["y"].to_numpy(), df["x1"].to_numpy() + df["treat"].to_numpy())


def test_set_outcome_requires_exactly_one_of_formula_or_fn():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError):
        gen.set_outcome()
    with pytest.raises(ValueError):
        gen.set_outcome(formula="x1", fn=lambda x1: x1, coefficients={"x1": 1.0})


def test_formula_without_coefficients_raises():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError):
        gen.set_outcome(formula="x1")


# ----------------------------------------------------------------------
# Treatment
# ----------------------------------------------------------------------


def test_treatment_zero_before_start_period():
    gen = PanelGenerator(n_entities=50, n_periods=10, seed=1)
    gen.add_treatment("treat", start_period=5, propensity=1.0)  # all entities treated eventually
    gen.set_outcome(fn=lambda treat: treat.astype(float), noise_std=0.0)
    df, _ = gen.generate()
    pre = df[df["period"] < 5]
    post = df[df["period"] >= 5]
    assert (pre["treat"] == 0).all()
    assert (post["treat"] == 1).all()


def test_treatment_fixed_per_entity_not_reshuffled_per_period():
    gen = PanelGenerator(n_entities=50, n_periods=10, seed=1)
    gen.add_treatment("treat", start_period=0, propensity=0.5)
    gen.set_outcome(fn=lambda treat: treat.astype(float), noise_std=0.0)
    df, _ = gen.generate()
    # Since start_period=0, treat should be constant within each entity across periods
    per_entity_nunique = df.groupby("entity_id")["treat"].nunique()
    assert (per_entity_nunique == 1).all()


def test_add_treatment_rejects_invalid_start_period():
    gen = PanelGenerator(n_entities=10, n_periods=5, seed=1)
    with pytest.raises(ValueError):
        gen.add_treatment("treat", start_period=5)  # must be < n_periods
    with pytest.raises(ValueError):
        gen.add_treatment("treat", start_period=-1)


def test_add_treatment_rejects_invalid_propensity():
    gen = PanelGenerator(n_entities=10, n_periods=5, seed=1)
    with pytest.raises(ValueError):
        gen.add_treatment("treat", propensity=1.5)
    with pytest.raises(ValueError):
        gen.add_treatment("treat", propensity=-0.1)


def test_add_treatment_rejects_unsupported_assignment():
    gen = PanelGenerator(n_entities=10, n_periods=5, seed=1)
    with pytest.raises(ValueError):
        gen.add_treatment("treat", assignment="staggered")


# ----------------------------------------------------------------------
# Variable declaration
# ----------------------------------------------------------------------


def test_add_variable_normal_and_uniform():
    gen = PanelGenerator(n_entities=1000, n_periods=1, seed=1)
    gen.add_variable("x_norm", dist="normal", mean=10.0, std=0.001)
    gen.add_variable("x_unif", dist="uniform", low=5.0, high=5.0001)
    gen.set_outcome(formula="x_norm", coefficients={"x_norm": 1.0}, noise_std=0.0)
    df, _ = gen.generate()
    assert df["x_norm"].mean() == pytest.approx(10.0, abs=0.01)
    assert (df["x_unif"] >= 5.0).all() and (df["x_unif"] <= 5.0001).all()


def test_add_variable_rejects_unsupported_dist():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError):
        gen.add_variable("x1", dist="banana")


def test_add_variable_rejects_negative_std():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError):
        gen.add_variable("x1", dist="normal", std=-1.0)


def test_add_variable_rejects_invalid_uniform_bounds():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError):
        gen.add_variable("x1", dist="uniform", low=5.0, high=1.0)


def test_duplicate_name_rejected_across_categories():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    gen.add_variable("x1")
    with pytest.raises(ValueError):
        gen.add_variable("x1")  # duplicate variable
    with pytest.raises(ValueError):
        gen.add_treatment("x1")  # collides with existing variable
    with pytest.raises(ValueError):
        gen.add_variable("entity_id")  # collides with reserved column name


def test_invalid_identifier_rejected():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    with pytest.raises(ValueError):
        gen.add_variable("not a valid name")


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_entities", [0, -1])
def test_constructor_rejects_invalid_n_entities(n_entities):
    with pytest.raises(ValueError):
        PanelGenerator(n_entities=n_entities, n_periods=5, seed=1)


@pytest.mark.parametrize("n_periods", [0, -1])
def test_constructor_rejects_invalid_n_periods(n_periods):
    with pytest.raises(ValueError):
        PanelGenerator(n_entities=5, n_periods=n_periods, seed=1)


def test_generate_without_outcome_raises():
    gen = PanelGenerator(n_entities=10, n_periods=5, seed=1)
    with pytest.raises(RuntimeError):
        gen.generate()


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------


def test_same_seed_byte_identical():
    df_a, _ = _basic_generator(seed=99).generate()
    df_b, _ = _basic_generator(seed=99).generate()
    pdt.assert_frame_equal(df_a, df_b)


def test_different_seed_different_data():
    df_a, _ = _basic_generator(seed=99).generate()
    df_b, _ = _basic_generator(seed=100).generate()
    assert not df_a["x1"].equals(df_b["x1"])


def test_builder_call_order_does_not_affect_variable_draws():
    """Adding treatment before vs. after a variable must not change that
    variable's drawn values, since each named stream is content-hashed."""
    gen_a = PanelGenerator(n_entities=30, n_periods=4, seed=7)
    gen_a.add_variable("x1")
    gen_a.add_treatment("treat", start_period=1)
    gen_a.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df_a, _ = gen_a.generate()

    gen_b = PanelGenerator(n_entities=30, n_periods=4, seed=7)
    gen_b.add_treatment("treat", start_period=1)  # added first this time
    gen_b.add_variable("x1")
    gen_b.set_outcome(formula="x1", coefficients={"x1": 1.0})
    df_b, _ = gen_b.generate()

    np.testing.assert_allclose(df_a["x1"].to_numpy(), df_b["x1"].to_numpy())
    np.testing.assert_allclose(df_a["treat"].to_numpy(), df_b["treat"].to_numpy())


# ----------------------------------------------------------------------
# GroundTruth contents
# ----------------------------------------------------------------------


def test_ground_truth_records_seed_and_versions():
    gen = _basic_generator(seed=123)
    _, truth = gen.generate()
    assert truth.seed == 123
    assert truth.spuriosity_version == "0.1.0"
    assert truth.numpy_version == np.__version__


def test_ground_truth_treatment_effect_ate_matches_coefficient():
    gen = _basic_generator()
    _, truth = gen.generate()
    assert truth.treatment_effect_ate == 3.0


def test_ground_truth_treatment_effect_none_without_treatment():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    _, truth = gen.generate()
    assert truth.treatment_effect_ate is None


# ----------------------------------------------------------------------
# __repr__
# ----------------------------------------------------------------------


def test_repr_includes_basic_shape_and_seed():
    """The repr is a debugging snapshot, not a round-trip. We just check
    that the key shape/seed info is in the first line."""
    gen = PanelGenerator(n_entities=100, n_periods=5, seed=99)
    r = repr(gen)
    assert "PanelGenerator" in r
    assert "n_entities=100" in r
    assert "n_periods=5" in r
    assert "seed=99" in r


def test_repr_shows_declared_variables_treatment_outcome():
    gen = _basic_generator(seed=7)
    r = repr(gen)
    # All three declared variables should appear
    assert "x1" in r
    assert "x2" in r
    assert "treat" in r
    # Treatment line
    assert "treatment:" in r
    # Outcome line (formula + coefficients + noise)
    assert "outcome:" in r
    assert "formula=" in r
    assert "noise_std" in r


def test_repr_shows_pathology_counts_by_type():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.set_outcome(formula="x1", coefficients={"x1": 1.0})
    gen.add_structural_break(period=0, target="y", kind="mean_shift", magnitude=1.0)
    gen.add_confounder(feature="x1", outcome="y", strength=0.5, observed=False)
    r = repr(gen)
    # The pathologies section should be present and name both types
    assert "pathologies (2):" in r
    assert "StructuralBreak" in r
    assert "Confounder" in r


def test_repr_handles_minimal_generator_gracefully():
    """A bare PanelGenerator with no variables / no treatment / no outcome
    should still produce a valid repr without raising."""
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    r = repr(gen)
    assert "PanelGenerator" in r
    assert "variables: <none>" in r
    assert "treatment: <none>" in r
    # The outcome is unset at this point -- a clear "<unset>" marker is the
    # whole point of the repr (tells you generate() will fail).
    assert "outcome: <unset" in r


def test_repr_shows_hte_when_present():
    gen = PanelGenerator(n_entities=10, n_periods=1, seed=1)
    gen.add_variable("x1")
    gen.add_treatment("treat", assignment="random", start_period=0)
    gen.set_outcome(
        formula="x1 + treat",
        coefficients={"x1": 0.0, "treat": 0.0, "Intercept": 0.0},  # treat coef 0 to avoid HTE warning
        noise_std=0.1,
    )
    gen.add_hte(treatment="treat", modifier="x1", formula="1.0 + 0.5*x1")
    r = repr(gen)
    assert "hte:" in r
    assert "treatment='treat'" in r
    assert "modifier='x1'" in r
