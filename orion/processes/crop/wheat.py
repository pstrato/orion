"""Wheat species: crop state plus process-scoped species parameters."""

from __future__ import annotations

from typing import Any

from orion.core.entity import Entity, entity
from orion.core.input import Input, Inputs
from orion.core.process import Process
from orion.core.state import State
from orion.core.variable import var
from orion.processes.crop.canopy_organ import CanopyOrgan, CanopyOrganInput, canopy_organ
from orion.processes.crop.crop import Crop, crop_bulk

WHEAT_K_LEAVES = 0.5
WHEAT_K_STEMS = 0.4
WHEAT_K_EARS = 0.6


class Stems(CanopyOrgan):
    """Wheat stems"""


class Leaves(CanopyOrgan):
    """Wheat leaves"""


class Ears(CanopyOrgan):
    """Wheat ears."""


@entity()
class Wheat(Crop):
    """Wheat crop."""

    leaves: Leaves
    """Wheat leaves."""
    stems: Stems
    """Wheat stems."""
    ears: Ears
    """Wheat ears."""

    @property
    def canopy(self) -> tuple[CanopyOrgan, ...]:
        return self.leaves, self.stems, self.ears

    @property
    def organs(self):
        return self.roots, self.leaves, self.stems, self.ears


@entity(meta=("k",))
class LeavesInput(CanopyOrganInput):
    """Leaves parameters."""


@entity(meta=("k",))
class StemsInput(CanopyOrganInput[Stems]):
    """Stems parameters."""

    def organ(self) -> Any:
        return Stems("stems", self, None, var("top", "m", 0, "Wheat stems"), var("bottom", "m", 0, "Wheat stems bottom position"), var("area_index", "m^2/m^2", 0, "Wheat stems area index"))


@entity(meta=("k",))
class EarsInput(CanopyOrganInput):
    """Ears parameters."""


@entity(meta=("k_leaves", "k_stems", "k_ears"))
class WheatInput(Entity, Input):
    """Wheat species: crop state with per-organ light-interception k."""

    leaves: LeavesInput = LeavesInput("leaves", k=WHEAT_K_LEAVES)
    stems: StemsInput = StemsInput("stems", k=WHEAT_K_STEMS)
    ears: EarsInput = EarsInput("ears", k=WHEAT_K_EARS)

    def states(self, inputs: Inputs) -> tuple[State, ...]:
        return Wheat(
            name="crop",
            constraint=None,
            leaves=canopy_organ("leaves", input=self.leaves),
            stems=canopy_organ("stems", input=self.stems),
            ears=canopy_organ("ears", input=self.ears),
            roots=crop_bulk("root"),
        )

    def processes(self, inputs: Inputs) -> tuple[Process, ...]:
        return ()


def wheat_input(name: str = "wheat") -> WheatInput:
    """Wheat input with species default extinction coefficients on canopy organs."""
    return WheatInput(name)
