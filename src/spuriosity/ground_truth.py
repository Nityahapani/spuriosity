"""
GroundTruth — the frozen record of a PanelGenerator's true data-generating
process, returned alongside every generated DataFrame.

Reproducibility contract: same seed + same pinned spuriosity/numpy versions
produces a byte-identical DataFrame. No cross-version guarantee — see
docs/design_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BreakInfo:
    period: int
    target: str
    kind: str
    magnitude: float


@dataclass(frozen=True)
class SelectionInfo:
    rule: str
    drop_prob: float


@dataclass(frozen=True)
class GroundTruth:
    true_coefficients: dict[str, float]
    break_points: list[BreakInfo] = field(default_factory=list)
    confounding_strength: dict[str, float] = field(default_factory=dict)
    true_cate: Optional[Callable[[float], float]] = None
    selection_mechanism: Optional[SelectionInfo] = None
    treatment_effect_ate: Optional[float] = None
    spuriosity_version: str = ""
    numpy_version: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError
