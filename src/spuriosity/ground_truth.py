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

    def __repr__(self) -> str:
        # Lead with the kind, since that's what you scan for first when debugging
        return f"BreakInfo(kind={self.kind!r}, period={self.period}, target={self.target!r}, magnitude={self.magnitude})"


@dataclass(frozen=True)
class SelectionInfo:
    """Record of the injected selection-bias mechanism."""

    rule: str
    drop_prob: float

    def __repr__(self) -> str:
        return f"SelectionInfo(rule={self.rule!r}, drop_prob={self.drop_prob})"


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

    def __repr__(self) -> str:
        """Compact, debugging-friendly summary of the ground truth.

        Designed for notebooks: one screen of text, every populated field
        visible at a glance, no truncation of the coefficient dict
        (these are usually 2–6 entries).
        """
        parts: list[str] = [f"GroundTruth(seed={self.seed}, spuriosity={self.spuriosity_version!r})"]

        if self.true_coefficients:
            coefs = ", ".join(f"{k!r}: {v}" for k, v in self.true_coefficients.items())
            parts.append(f"  true_coefficients: {{{coefs}}}")
        else:
            parts.append("  true_coefficients: <empty>")

        if self.break_points:
            parts.append(f"  break_points: {len(self.break_points)} ({[b.kind for b in self.break_points]})")
        if self.confounding_strength:
            parts.append(f"  confounding_strength: {self.confounding_strength}")
        if self.selection_mechanism is not None:
            sm = self.selection_mechanism
            parts.append(f"  selection_mechanism: rule={sm.rule!r}, drop_prob={sm.drop_prob}")
        if self.treatment_effect_ate is not None:
            parts.append(f"  treatment_effect_ate: {self.treatment_effect_ate:.4f}")
        parts.append(f"  has_true_cate: {self.true_cate is not None}")

        return "\n".join(parts)

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
