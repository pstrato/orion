"""Plant root state — triangular density from the soil surface down to bottom."""

from __future__ import annotations

from orion.core.constant import Constant
from orion.core.entity import entity
from orion.processes.crop.organ import Organ, OrganInput
from orion.processes.crop.shape import Shape


@entity()
class Roots(Organ):
    """Crop roots."""


class RootsInput(OrganInput):
    """Root system parameters."""

    shape: Constant[Shape]
    """Roots shape."""

    def organ(self) -> Roots:
        return Roots("roots", None)
