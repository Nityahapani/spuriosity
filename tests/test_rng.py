"""Tests for spuriosity._rng.

The critical property under test is order-independence: the same set of
named streams must produce byte-identical draws regardless of the order in
which they are first requested. An earlier draft of this module violated
that property for dynamic (non-canonical) names by assigning spawn-tree
slots in request order rather than by a stable hash of the name; these
tests exist specifically to catch a regression of that bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from spuriosity._rng import RNGManager


def test_same_name_same_seed_identical_draws():
    m1 = RNGManager(42)
    m2 = RNGManager(42)
    a = m1.child("confounder:x1").normal(size=5)
    b = m2.child("confounder:x1").normal(size=5)
    assert np.array_equal(a, b)


def test_dynamic_name_order_independence():
    """Regression test: requesting the same set of dynamic names in a
    different order must not change the stream assigned to any one name."""
    m_first = RNGManager(42)
    draw_when_requested_first = m_first.child("confounder:x1").normal(size=5)

    m_third = RNGManager(42)
    _ = m_third.child("selection_bias:rule1")
    _ = m_third.child("structural_break:period20")
    draw_when_requested_third = m_third.child("confounder:x1").normal(size=5)

    assert np.array_equal(draw_when_requested_first, draw_when_requested_third)


def test_canonical_stream_order_independence():
    m_a = RNGManager(42)
    base_a = m_a.child("base_variables").normal(size=3)
    _ = m_a.child("some_dynamic_thing")

    m_b = RNGManager(42)
    _ = m_b.child("some_dynamic_thing")  # dynamic requested before canonical
    base_b = m_b.child("base_variables").normal(size=3)

    assert np.array_equal(base_a, base_b)


def test_distinct_names_give_distinct_streams():
    m = RNGManager(42)
    s1 = m.child("confounder:x1").normal(size=5)
    s2 = m.child("confounder:x2").normal(size=5)
    assert not np.array_equal(s1, s2)


def test_distinct_seeds_give_distinct_streams():
    m1 = RNGManager(42)
    m2 = RNGManager(43)
    d1 = m1.child("confounder:x1").normal(size=5)
    d2 = m2.child("confounder:x1").normal(size=5)
    assert not np.array_equal(d1, d2)


def test_canonical_name_resolves_consistently_as_dynamic_lookup():
    """'base_variables' is a canonical name; requesting it should always
    resolve to the same fixed canonical stream, not a hashed dynamic one."""
    m1 = RNGManager(42)
    m2 = RNGManager(42)
    assert np.array_equal(
        m1.child("base_variables").normal(size=3),
        m2.child("base_variables").normal(size=3),
    )


def test_repeated_child_calls_return_same_generator_instance():
    m = RNGManager(42)
    g_a = m.child("confounder:x1")
    g_b = m.child("confounder:x1")
    assert g_a is g_b


def test_repeated_draws_continue_the_stream_not_reset():
    m = RNGManager(42)
    gen = m.child("confounder:x1")
    draw1 = gen.normal(size=3)
    draw2 = gen.normal(size=3)
    assert not np.array_equal(draw1, draw2)


@pytest.mark.parametrize("bad_seed", ["not an int", 3.14, None, [1, 2, 3]])
def test_non_int_seed_raises_type_error(bad_seed):
    with pytest.raises(TypeError):
        RNGManager(bad_seed)


def test_bool_seed_rejected_explicitly():
    """bool is technically a subclass of int in Python; reject it anyway
    to avoid RNGManager(True) silently behaving like RNGManager(1)."""
    with pytest.raises(TypeError):
        RNGManager(True)


def test_repr_contains_seed_and_dynamic_count():
    m = RNGManager(1)
    _ = m.child("a")
    _ = m.child("b")
    r = repr(m)
    assert "seed=1" in r
    assert "dynamic_streams_assigned=2" in r
