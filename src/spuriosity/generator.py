"""
PanelGenerator — the main builder for synthetic panel datasets.

Holds entity/period structure, base variables, the outcome DGP (via patsy
formula or callable), treatment assignment, heterogeneous treatment effects,
and a stack of pathologies to apply before generation.

See docs/design_spec.md for the full API design.
"""

from __future__ import annotations


class PanelGenerator:
    """Builder for synthetic panel data with a known data-generating process.

    Not yet implemented — see docs/design_spec.md for target API.
    """

    def __init__(self, n_entities: int, n_periods: int, seed: int) -> None:
        raise NotImplementedError

    def add_variable(self, name: str, dist: str, **kwargs) -> "PanelGenerator":
        raise NotImplementedError

    def add_treatment(self, name: str, assignment: str, **kwargs) -> "PanelGenerator":
        raise NotImplementedError

    def set_outcome(self, formula: str | None = None, fn=None, noise_std: float = 1.0) -> "PanelGenerator":
        raise NotImplementedError

    def add_hte(self, treatment: str, modifier: str, formula: str) -> "PanelGenerator":
        raise NotImplementedError

    def add_structural_break(self, period: int, target: str, kind: str, magnitude: float) -> "PanelGenerator":
        raise NotImplementedError

    def add_confounder(self, feature: str, outcome: str, strength: float, observed: bool = False) -> "PanelGenerator":
        raise NotImplementedError

    def add_selection_bias(self, rule: str, drop_prob: float) -> "PanelGenerator":
        raise NotImplementedError

    def validate_combo(self) -> None:
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError
