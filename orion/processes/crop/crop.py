"""Shared crop state updated by crop processes."""

from __future__ import annotations

from orion.core.entity import entity
from orion.core.state import State
from orion.processes.crop.canopy_organ import CanopyOrgan
from orion.processes.crop.organ import Organ
from orion.processes.crop.roots import Roots


@entity()
class Crop(State):
    """Shared crop state: species canopy, root, and bulk quantities."""

    roots: Roots
    """Crop root."""

    @property
    def canopy(self) -> tuple[CanopyOrgan, ...]:
        """Above-ground organs that intercept light (species-specific)."""
        raise NotImplementedError()

    @property
    def organs(self) -> tuple[Organ, ...]:
        """All organs of the plan."""
        raise NotImplementedError()
