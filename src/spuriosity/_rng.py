"""
RNGManager — order-independent RNG sub-streams for reproducible generation.

A single global `np.random.default_rng(seed)`, drawn from sequentially across
pathologies, is fragile: the resulting stream depends on the *order* in which
pathologies happen to be added, not just which ones are present. Adding
`Confounder` before `SelectionBias` vs. after would silently produce a
different dataset for the same logical spec.

`RNGManager` avoids this by spawning independent child streams from a root
`numpy.random.SeedSequence`, keyed by a stable *name* rather than by draw
order or request order. Each name deterministically maps to a spawn-tree
position via a stable hash of the name itself (not the order it happens to
be requested in), so two `PanelGenerator` specs with the same seed and the
same named components produce byte-identical results regardless of the
order in which `.add_*()` calls were made or the order streams are first
accessed.

Reproducibility contract: same seed + same pinned `spuriosity`/`numpy`
versions -> byte-identical streams. No cross-version guarantee (numpy's
bit generator / SeedSequence spawning behavior is not guaranteed stable
across numpy releases). See docs/design_spec.md.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Canonical, fixed ordering of well-known stream names. New well-known names
# should be appended here, never inserted in the middle -- doing so would
# reassign existing names to different spawn-tree positions and break
# reproducibility for existing specs pinned to an older spuriosity version.
_CANONICAL_STREAMS: tuple[str, ...] = (
    "base_variables",
    "treatment_assignment",
    "outcome_noise",
)


def _name_to_spawn_key(name: str) -> int:
    """Deterministically map an arbitrary string name to a 63-bit
    non-negative integer, used as extra `SeedSequence` entropy.

    This must depend only on the name's *content*, never on request order,
    insertion order, or Python's randomized string hashing (`hash()` is
    salted per-process and is NOT used here for exactly that reason).
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # Take the first 8 bytes as a big-endian unsigned int, then mask to 63
    # bits so it's safely representable and non-negative across platforms.
    return int.from_bytes(digest[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


class RNGManager:
    """Owns a root SeedSequence and hands out named, independent child
    `numpy.random.Generator` sub-streams.

    Well-known streams (see `_CANONICAL_STREAMS`) always occupy the same
    fixed spawn-tree positions. Any other ("dynamic") name is mapped
    deterministically to its own spawn-tree position by hashing the name's
    content (`_name_to_spawn_key`) -- never by the order in which it is
    first requested. This guarantees that a `PanelGenerator` built by adding
    the same named components in a different order produces byte-identical
    streams.

    Note that repeatedly *drawing* from a returned generator is stateful,
    like any `numpy.random.Generator` -- `.child(name)` should be called
    once per component and the returned generator reused, not re-requested
    expecting a fresh stream.
    """

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.seed = seed
        self._root = np.random.SeedSequence(seed)

        # Spawn canonical streams eagerly, in fixed order, so their spawn-tree
        # positions never depend on what the caller does.
        canonical_children = self._root.spawn(len(_CANONICAL_STREAMS))
        self._canonical: dict[str, np.random.Generator] = {
            name: np.random.default_rng(child)
            for name, child in zip(_CANONICAL_STREAMS, canonical_children)
        }

        # Dynamic names get their own SeedSequence, derived from the root
        # seed plus a deterministic hash of the name -- entirely independent
        # of request order. Cached on first request so repeated `.child()`
        # calls for the same name return the same (stateful) generator
        # rather than resetting it.
        self._dynamic_assigned: dict[str, np.random.Generator] = {}

    def child(self, name: str) -> np.random.Generator:
        """Return the named child generator, creating it on first request.

        Canonical names (see `_CANONICAL_STREAMS`) resolve to a fixed
        spawn-tree position. Any other name is mapped deterministically to
        its own independent stream via a content hash of the name -- the
        result does not depend on request order.
        """
        if name in self._canonical:
            return self._canonical[name]

        if name not in self._dynamic_assigned:
            spawn_key = _name_to_spawn_key(name)
            child_seq = np.random.SeedSequence([self.seed, spawn_key])
            self._dynamic_assigned[name] = np.random.default_rng(child_seq)

        return self._dynamic_assigned[name]

    def __repr__(self) -> str:
        return f"RNGManager(seed={self.seed!r}, dynamic_streams_assigned={len(self._dynamic_assigned)})"
