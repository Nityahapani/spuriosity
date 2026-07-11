"""
GroundTruth — the frozen record of a PanelGenerator's true data-generating
process, returned alongside every generated DataFrame.

Reproducibility contract: same seed + same pinned spuriosity/numpy versions
produces a byte-identical DataFrame. No cross-version guarantee — see
docs/design_spec.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BreakInfo:
    """Record of a single injected structural break."""

    period: int
    target: str
    kind: str
    magnitude: float


@dataclass(frozen=True)
class SelectionInfo:
    """Record of the injected selection-bias mechanism."""

    rule: str
    drop_prob: float


@dataclass(frozen=True)
class GroundTruth:
    """The true data-generating process behind a generated panel dataset.

    `true_cate`, if set, is a callable excluded from serialization (it isn't
    JSON-representable) — `to_dict()`/`to_json()` note its presence via
    `has_true_cate` instead of attempting to serialize the function itself.

    Reproducibility contract: same `seed` + same pinned `spuriosity` and
    `numpy` versions produces a byte-identical generated DataFrame. No
    cross-version guarantee is made — see docs/design_spec.md.
    """

    true_coefficients: dict[str, float]
    break_points: list[BreakInfo] = field(default_factory=list)
    confounding_strength: dict[str, float] = field(default_factory=dict)
    true_cate: Optional[Callable[[float], float]] = None
    selection_mechanism: Optional[SelectionInfo] = None
    treatment_effect_ate: Optional[float] = None
    spuriosity_version: str = ""
    numpy_version: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict. `true_cate`, being a callable, is not
        included directly; its presence is flagged via `has_true_cate`."""
        d: dict[str, Any] = {
            "true_coefficients": dict(self.true_coefficients),
            "break_points": [asdict(b) for b in self.break_points],
            "confounding_strength": dict(self.confounding_strength),
            "has_true_cate": self.true_cate is not None,
            "selection_mechanism": (
                asdict(self.selection_mechanism) if self.selection_mechanism is not None else None
            ),
            "treatment_effect_ate": self.treatment_effect_ate,
            "spuriosity_version": self.spuriosity_version,
            "numpy_version": self.numpy_version,
            "seed": self.seed,
        }
        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string via `to_dict()`."""
        return json.dumps(self.to_dict(), indent=indent)
